import numpy as np
from numpy import array as A
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon
from flightTools import thrust_parse


class FlightAnimation:
    def __init__(self, flight, filename, debug=False):
        """
        Animates a given completed flight, saves at filename
        """
        self.flight = flight
        self.animate(filename, debug)

    def animate(self, filename, debug):

        hull_colour = '#3b4049'
        accent_colour = '#ffffff'
        thrust_colour = '#e28b44'

        # Intitialise figure
        booster_width = self.flight.rocket.width * 0.3
        frame_scale = self.flight.sim_scale / 2
        self.leg_height = self.flight.rocket.height * 0.2
        self.fig, self.ax = subplots(figsize=(8, 8), dpi=150)
        self.ax.imshow(A([np.linspace(0.0, 1.0, 101)] * 101).T, cmap=plt.get_cmap('Blues'),
                       vmin=0.0, vmax=2.0, origin='lower', extent=[-frame_scale, frame_scale, 0, 2 * frame_scale])
        self.ax.set(xlim=[- frame_scale, frame_scale], ylim=[-10.0, 2 * frame_scale],
                    ylabel='Altitude (m)', xlabel='', xticks=[])
        self.ax.grid(False)

        # Find centre of rocket
        self.pos_x_0 = self.flight.position[0][0]
        self.pos_y_0 = self.flight.position[0][1]

        # Flight data
        telemetry = 'Status: {:>7s}\nT Vel:    {:>7.2f}\nH Vel:    {:>7.2f}\nV Vel:    {:>7.2f}\nA Vel:    {:>7.2f}\nScore:    {:>7.2f}'.format(
            self.flight.status_string(0), norm(self.flight.velocity[0]),
            self.flight.velocity[0][0], self.flight.velocity[0][1],
            self.flight.angular_v[0],
            self.flight.score[0])
        self.telemetry = self.ax.text(
            -0.9 * frame_scale,
            1.9 * frame_scale,
            telemetry, ha='left', va='top',
            fontname='monospace', alpha=0.8)

        # Body of rocket
        self.body = Rectangle(
            (self.pos_x_0 - self.flight.rocket.width / 2,
             self.pos_y_0 - (self.flight.rocket.height / 2 - self.leg_height)),
            self.flight.rocket.width,
            self.flight.rocket.height - self.leg_height,
            angle=0.0,
            color=hull_colour, zorder=2)

        # Main thruster
        self.main = Ellipse(
            (self.pos_x_0,
             self.pos_y_0 - (self.flight.rocket.height / 2 - self.leg_height)),
            self.flight.rocket.width * 0.8,
            2 * self.leg_height * thrust_parse(self.flight.throttle[0], mode=self.flight.mode)[0],
            angle=0.0,
            color=thrust_colour, zorder=1)

        # L Booster
        self.LBooster = Ellipse(
            (self.pos_x_0 - self.flight.rocket.width / 2,
             self.pos_y_0 + (0.5 * self.flight.rocket.height - 2 * booster_width)),
            self.flight.rocket.width * 2 * thrust_parse(self.flight.throttle[0], mode=self.flight.mode)[1],
            booster_width * 2,
            color=thrust_colour)

        # L Booster
        self.RBooster = Ellipse(
            (self.pos_x_0 + self.flight.rocket.width / 2,
             self.pos_y_0 + (0.5 * self.flight.rocket.height - 2 * booster_width)),
            self.flight.rocket.width * 2 * thrust_parse(self.flight.throttle[0], mode=self.flight.mode)[2],
            booster_width * 2,
            color=thrust_colour)

        # Legs
        self.legs = Polygon(A([[self.pos_x_0 - (self.flight.rocket.width / 2 + 0.5 * self.leg_height),
                                self.pos_y_0 - self.flight.rocket.height / 2],
                               [self.pos_x_0,
                                self.pos_y_0 - 0.5 * self.leg_height],
                               [self.pos_x_0 + (self.flight.rocket.width / 2 + 0.5 * self.leg_height),
                                self.pos_y_0 - self.flight.rocket.height / 2]
                               ]),
                            closed=False, fill=False, edgecolor=hull_colour, zorder=1,
                            lw=2.0)

        # Landing area
        self.ground = Rectangle((- frame_scale, -10), 2 * frame_scale, 10, color='#4c91ad', zorder=0)
        self.base = Rectangle((-self.flight.base_size, - 5), 2 * self.flight.base_size, 5, color='#bdbfc1', zorder=1)

        # Draw objects to canvas
        self.patches = [self.body, self.main,
                        self.LBooster, self.RBooster, self.legs]

        ts = self.ax.transData
        # Translate and rotate everything else
        tr = Affine2D().rotate_around(self.flight.position[0][0],
                                      self.flight.position[0][1],
                                      self.flight.angle[0])
        transform = tr + ts

        if debug:
            self.com = self.ax.plot([self.pos_x_0], [self.pos_y_0], 'x', ms=20, lw=2.0)
            self.ax.plot(self.flight.position[:, 0], self.flight.position[:, 1], '--k', alpha=0.8)
            b_h = self.flight.rocket.height
            b_w = self.flight.rocket.height
            self.bbox = Rectangle((self.pos_x_0 - b_w / 2,
                                   self.pos_y_0 - b_h / 2),
                                  b_w, b_h, edgecolor='k', fill=False, lw=2.0)
            self.ax.add_artist(self.bbox)

        for p in self.patches:
            self.ax.add_artist(p)
            p.set_transform(transform)

        self.ax.add_artist(self.ground)
        self.ax.add_artist(self.base)

        self.fig.tight_layout()

        # Add an extra second to the end of the animation
        extra_frames = int(1.0 / self.flight.simulation_resolution)

        # Animate the plot according to teh update function (below)
        movie = FuncAnimation(self.fig, self.update_animation,
                              interval=(1000 * self.flight.simulation_resolution),
                              frames=(len(self.flight.position + extra_frames)),
                              fargs=[debug])
        movie.save(filename)

    def update_animation(self, i, debug):

        #  Get transformation data
        ts = self.ax.transData
        dx = self.flight.position[i][0] - self.pos_x_0
        dy = self.flight.position[i][1] - self.pos_y_0

        if debug:
            self.com[0].set_data([self.flight.position[i][0]], [self.flight.position[i][1]])
            b_h = self.flight.rocket.height * np.abs(np.cos(self.flight.angle[i])) + self.flight.rocket.width * np.abs(
                np.sin(self.flight.angle[i]))
            b_w = self.flight.rocket.height * np.abs(np.sin(self.flight.angle[i])) + self.flight.rocket.width * np.abs(
                np.cos(self.flight.angle[i]))
            self.bbox.xy = (self.flight.position[i][0] - b_w / 2, self.flight.position[i][1] - b_h / 2)
            self.bbox.set_height(b_h)
            self.bbox.set_width(b_w)

        # Update text
        telemetry = 'Status:   {:>7s}\nT Vel:    {:>7.2f}\nH Vel:    {:>7.2f}\nV Vel:    {:>7.2f}\nA Vel:    {:>7.2f}\nScore:    {:>7.2f}'.format(
            self.flight.status_string(i), norm(self.flight.velocity[i]),
            self.flight.velocity[i][0], self.flight.velocity[i][1],
            self.flight.angular_v[i],
            self.flight.score[:(i + 1)].sum())
        self.telemetry.set_text(telemetry)

        # Thrusters on/off
        self.main.height = 2 * self.leg_height * thrust_parse(self.flight.throttle[i], mode=self.flight.mode)[0]
        self.LBooster.width = self.flight.rocket.width * 2 * \
                              thrust_parse(self.flight.throttle[i], mode=self.flight.mode)[1]
        self.RBooster.width = self.flight.rocket.width * 2 * \
                              thrust_parse(self.flight.throttle[i], mode=self.flight.mode)[2]

        # Translate and rotate everything else
        tr = Affine2D().translate(dx, dy).rotate_around(self.flight.position[i][0],
                                                        self.flight.position[i][1],
                                                        self.flight.angle[i])

        transform = tr + ts
        for p in self.patches:
            p.set_transform(transform)


def flight_data_plot(flight, save=''):
    """
    Plots various data for a given flight
    """

    plt.style.use('ggplot')
    fig, ax = plt.subplots(6, 1)

    labels = ['Position (m)', 'Velocity (ms$^{-1}$)', 'Acceleration (ms$^{-2}$)',
              'Fuel Used (%)', 'Throttle (%)', 'Score']

    y_axis = [flight.position[:, 1], flight.velocity[:, 1],
              flight.acceleration[:, 1],
              100.0 * (flight.mass - flight.rocket.hull_mass) / flight.rocket.fuel_mass,
              100.0 * flight.throttle, np.cumsum(flight.score)]

    for i, a in enumerate(ax):
        a.plot(flight.time, y_axis[i], color=('C%d' % i))
        a.set_ylabel(labels[i])
        a.set(xlim=[0, flight.time.max()])
        if i < 5:
            a.set_xticklabels([])
    ax[4].set_xlabel('Time (s)')

    fig.subplots_adjust(hspace=0.05)
    fig.set_size_inches(10, 12)

    if save:
        fig.savefig(save)

    return fig
