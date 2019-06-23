from flightTools import Flight, thrust_parse
from trainingTools import FlightController
from numpy.linalg import norm
from numpy import array as A
from keras.layers import Dense
from keras.models import Sequential
from plotTools import FlightAnimation, flight_data_plot
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

# Say how many inputs and outputs we need
input_size, output_size = 4, 6

model = Sequential()

# First layer: two inputs and two outputs
model.add(Dense(12, input_shape=(input_size,), activation='tanh'))

# Second layer: two inputs (keras works that out from the previous layer) and two outputs, linear act. function
model.add(Dense(12, activation='linear'))

model.add(Dense(output_size, activation='linear'))

# Finally compile the model ready for training
model.compile(optimizer='adam', loss='mse')


#some constants
we1 = 1.1
we2 = 1/13
we3 = 0.2
we4 = 2
we5 = 2

# This simply rewards moving closer to the launchpad
# The score is just the fractional decrease in distance, compared to previous timestep
def reward_function(flight):
    # Initialise
    reward = 0.0

    # Give a reward equal to the (negative) difference in position, moving lower gives more reward
    reward -= we1*(flight.position[-1][1] - flight.position[-2][1]) + we2 * (np.abs(flight.velocity[-1][1]) - flight.rocket.impact_velocity) \
         + we4 * (flight.time[-1]-flight.time[-2]) + we5*(np.abs(flight.position[-1][0]) - np.abs(flight.position[-2][0]))

    if flight.throttle[-1] > 0:
        reward += we3 * (flight.mass[-1] - flight.mass[-2])

    if flight.status[-1] == 2:
        reward += 100.0

    # Give a penalty if the rocket crashes
    elif abs(flight.status[-1]) == 1:
        reward -= 100.0

    return reward
    # If the rocket lands successfully give an extra bonus

flight_controller = FlightController(input_size, output_size, model)

flight = Flight(flight_controller=flight_controller,
                reward_function=reward_function, mode=0)

flight.run()


# The mode we're using
mode = 1

# The number of flights we'll run
episodes = 200000

# The number of moves (not flights) on which to train the model in batches
batch_size = 50

# The flight controller we're going to train
flight_controller = FlightController(input_size, output_size, model)

#flight_controller.load('./weights1/069500.h5')

# The reward function we will use
reward_function = reward_function

landingcounter = 0

landA = []

scoreA = []

episodeA = [i for i in range(episodes)]

for i in range(episodes+1):

    # Initialise a new flight. This randomises the initial conditions each time
    flight = Flight(flight_controller=flight_controller,
                    reward_function=reward_function, mode=mode)

    # Get the initital state vector
    done, total_reward = False, 0
    state = A([flight.state_vector(mode)])

    # Update the flight until it crashes
    while not done:

        # Get the action from the flight controller
        action = flight_controller(state)

        # Update the flight and get the reward and the new state
        next_state, reward, done = flight.update(action)

        # Add the reward from the previous step to the total
        total_reward += reward

        # Transform the state vector (Keras only takes 2D)
        next_state = np.reshape(next_state, [1, input_size])

        # Commit this iteration to the memory
        flight_controller.remember(state, action, reward, next_state, done)
        state = next_state

        # When the flight is over...
        if done:
            
            scoreA.append(total_reward)
            
            if flight.status[-1] == 2:
                landingcounter += 1    
                
            # Print and save the results every 50th flight (maybe change this)
            if i % 500 == 0:
                print("\r", "{:6d}/{:6d}, {:8s}, score: {: 3.2f}, e: {:3.2f}".format(
                    i, episodes, flight.status_string(), total_reward, flight_controller.epsilon
                ))
                print(landingcounter, '/ 500')

                plt.close('all')
                
                landA.append(landingcounter)
                
                # Make an animation of the flight so we can check progress
                FlightAnimation(flight, './plots12/%06d.mp4' % i)

                #Make plot
                flight_data_plot(flight, './graphs12/%06d.png' % i)

                # Save the weights so we can start training from here
                flight_controller.save('./weights12/%06d.h5' % i)

                landingcounter = 0
                
                break
            
            with open('./hello/score.txt', 'w') as f:
                f.write(str(scoreA))
                
            with open('./hello/land.txt','w') as g:
                g.write(str(landA))
            
    # When there are enough saved moves, start training each
    # time a new flight completes
    if len(flight_controller.memory) > batch_size:
        flight_controller.replay(batch_size)
        
#%% 
import scipy as sp
from scipy.optimize import curve_fit

plt.style.use('ggplot')
fig1, ax1 = plt.subplots(dpi = 200)        
ax1.plot([i for i in range(200000)],scoreA[:200000],'.',ms=0.1)
ax1.set_xlabel(r'Episode / Thousand')
ax1.set_ylabel(r'Score per Episode')
ax1.set_yticks([-800+50*i for i in range(23)],minor=True)
ax1.set_xticks([0,25e3,50e3,75e3,100e3,125e3,150e3,175e3,200e3])
ax1.set_xticklabels(['0','25','50','75','100','125','150','175','200'])
ax1.grid(b=True,axis='both',which='major',linestyle=':')
#plt.title(r'Graph to Show Total Score Per Episode of Training (Mode 1)')

fig2, ax2 = plt.subplots(dpi=1200)

ax2.plot([500*i for i in range(len(landA[:-1]))],landA[:-1],'x')

ax2.set_xlabel(r'Episode / Thousand')
ax2.set_ylabel(r'Mean Number of Landings',color='#e2452d')
ax2.grid(b=True,axis='both',which='major',linestyle=':')
ax2.tick_params(axis='y',which='both',color='#e2452d',labelcolor='#e2452d')

ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

epi1 = sp.array([500*i for i in range(len(landA[:-1]))])
land1 = sp.array([i/500 for i in landA[:-1]])

def fsigmoid(x, a, b, c, d):
    return a+ b/ (1.0 + np.exp(c-d*x))

popt, pcov = curve_fit(fsigmoid, epi1, land1,p0=[0,1,60,0.001])

x = sp.arange(200000)
y = fsigmoid(x, *popt)

ax3.plot(x[:125000],y[:125000],color='#0180c3')
ax3.plot(x[125000:200000],y[125000:200000],color='#0180c3',linestyle=':')

ax3.set_yticks([0.2*i for i in range(6)])
ax2.set_yticks([100*i for i in range(6)])
ax2.set_ylim(-25,500)
ax3.set_ylim(-0.05,1)


ax2.set_xticks([0,25e3,50e3,75e3,100e3,125e3,150e3,175e3,200e3])
ax2.set_xticklabels(['0','25','50','75','100','125','150','175','200'])

ax3.grid(b=True,axis='both',which='major',linestyle=':')

ax2.set_yticks([25*i for i in range(20)],minor=True)
ax3.set_yticks([0.05*i for i in range(20)],minor=True)

ax3.set_ylabel('Probability of Landing', color = '#0180c3')  # we already handled the x-label with ax1
ax3.tick_params(axis='y', which='both', labelcolor = '#0180c3', color = '#0180c3')
#plt.title(r'Graph to Show the Number and Probability of Landings (Mode 1)',pad=12)