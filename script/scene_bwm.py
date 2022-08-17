import os
from collections import OrderedDict

import numpy as np
from perlin_noise import PerlinNoise

from interface import ibpy
from interface.ibpy import Vector, Quaternion, change_emission_to
from objects.arrow import Arrow
from objects.bobject import BObject
from objects.circle import CircleArc, RightAngle
from objects.coordinate_system import CoordinateSystem
from objects.curtain import Curtain
from objects.curve import Curve
from objects.cylinder import Cylinder
from objects.derived_objects.flag import Flag
from objects.derived_objects.info_panel import InfoPanel
from objects.derived_objects.p_arrow import PArrow
from objects.derived_objects.pencil import Pencil
from objects.derived_objects.person_with_cape import PersonWithCape
from objects.derived_objects.pin import Pin
from objects.derived_objects.wall_with_door import WallWithDoor
from objects.digital_number import DigitalNumber
from objects.display import Display
from objects.ellipse import Ellipse
from objects.empties import EmptyCube
from objects.eraser.explosion import Explosion
from objects.eraser.fields import Turbulence, Wind, Force
from objects.floor import Floor
from objects.geometry.Person import Person
from objects.light.light import SpotLight

from objects.logo import LogoPreImage
from objects.number_line import NumberLine
from objects.plane import Plane
from objects.polygon import Polygon
from objects.geometry.sphere import Sphere, StackOfSpheres
from objects.rope import Rope
from objects.svg_bobject import SVGBObject
from objects.svg_objects.foot_prints import Paws
from objects.tex_bobject import SimpleTexBObject, TexBObject
from perform.scene import Scene
from tile_puzzle.tile_puzzle import SliderPuzzle
from tools.images import ImageCreator
from utils.constants import DOWN, RIGHT, UP, FRAME_RATE, SVG_DIR, DEFAULT_ANIMATION_TIME
from utils.utils import print_time_report, z2str, z2p, re, p2z, to_vector, retrieve_name, get_from_dictionary


# some functions created on demand


def riemann(x, y):
    """
    Projection onto the Riemann sphere
    :param x:
    :param y:
    :return:
    """
    d = 1 + x ** 2 + y ** 2
    X = x / d
    Y = y / d
    Z = (d - 1) / d
    return [X, Y, Z]


def riemann_r(x, y, r):
    """
    Projection onto the Riemann sphere with radius r
    :param x:
    :param y:
    :param r:
    :return:
    """
    r2 = r * r
    d = x * x + y * y + r2
    X = 2 * r2 * x / d
    Y = 2 * r2 * y / d
    Z = -r * (d - 2 * r2) / d
    return [X, Y, Z]


def flatten(nested_color_list):
    return [col for sub_list in nested_color_list for col in sub_list]


def z2loc(coords, z):
    return coords.coords2location(z2p(z))


def line1(l):
    zb = 1 + 2j
    za = -1 + 1j
    return za + l * (zb - za)


def line2(l):
    zc = 4 + 1j
    zd = 5 - 1j
    return zc + l * (zd - zc)


def play_problem1(t0, duration):
    stack_scale = [0.27, 0.27, 0.27]
    stack = StackOfSpheres(radius=0.5, dim=7, color='joker', smooth=2, name="FullStack", scale=stack_scale,
                           location=[0, 2.8175, 0.0927])
    stack.appear(count=len(stack.spheres), begin_time=t0, transition_time=duration / 10)

    t0 += duration / 10
    # alice and bob appear
    person_scale = [0.29, 0.29, 0.29]
    b_obj_alice = BObject.from_file('Person', color='drawing', location=[-3.5, 2.8, 0.43], name="Alice",
                                    scale=person_scale)
    b_obj_alice.appear(begin_time=t0, transition_time=duration / 20)
    b_obj_alice.move(direction=[2.12, 0, 0], begin_time=t0 + duration / 20, transition_time=duration / 20)

    b_obj_bob = BObject.from_file('Person', color='important', location=[3.5, 2.8, 0.43], name="Bob",
                                  scale=person_scale)
    b_obj_bob.appear(begin_time=t0, transition_time=duration / 20)
    b_obj_bob.move(direction=[-2.12, 0, 0], begin_time=t0 + duration / 20, transition_time=duration / 20)

    t0 += duration / 10
    # alice and bob take their shares
    sequence = [50, 20, 20, 20, 3, 1, 1]

    alice = 0
    bob = 0
    for i in range(len(sequence)):
        if i % 2 == 0:
            alice += sequence[i]
        else:
            bob += sequence[i]

    stack_a = StackOfSpheres(radius=0.5, number_of_spheres=alice, color='drawing', smooth=2,
                             location=[-2.6894, 2.8364, 0.116], name="StackAlice", scale=stack_scale)
    stack_b = StackOfSpheres(radius=0.5, number_of_spheres=bob, color='important', smooth=2,
                             location=[2.4894, 2.8364, 0.116], name="StackBob", scale=stack_scale)

    dt = 8 * duration / 10 / len(sequence)
    for i in range(len(sequence)):
        stack.appear(incr=-sequence[i], begin_time=t0, transition_time=dt / 3)
        if i % 2 == 0:
            stack_a.appear(incr=sequence[i], begin_time=t0 + dt / 3, transition_time=dt / 3)
        else:
            stack_b.appear(incr=sequence[i], begin_time=t0 + dt / 3, transition_time=dt / 3)
        t0 += dt


def play_problem4(t0, duration):
    # seq = np.random.randint(0,4,(50))
    # print(seq)
    seq = [3, 3, 0, 1, 3, 1, 1, 0, 1, 1, 0, 1, 1, 3, 1, 2, 3, 3, 1, 1, 1, 2, 3, 1, 1, 1, 1, 3, 2, 0, 2, 0, 0, 2, 2,
           1, 0, 2, 1, 2, 1, 1, 2, 0, 1, 3, 1, 0, 0, 1, 1, 1, 2, 0, 1, 3, 1, 0, 0, 1]
    colors = ['example', 'important', 'drawing', 'text']
    dx = 0.5
    dy = 0.5
    x0 = -2.7
    y0 = -3.8
    dt = duration * 0.8 / 50
    for i in range(5):
        for j in range(12):
            sphere = Sphere(r=0.1, location=[x0 + dx * j, y0 - dy * i, 0], color=colors[seq[i * 12 + j]])
            sphere.grow(begin_time=t0, transition_time=dt)
            t0 += dt

    polygons = [[(0, 0), (1, 0), (1, 1)], [(0, 2), (2, 2), (2, 4)], [(5, 2), (7, 2), (5, 4)],
                [(9, 1), (9, 2), (10, 2)]]
    colors = ['text', 'important', 'example', 'drawing']
    dt = duration * 0.2 / len(polygons)
    p_count = 0
    for polygon, color in zip(polygons, colors):
        vertices = []
        for tpl in polygon:
            vertices.append(Vector((x0 + tpl[0] * dx, y0 - tpl[1] * dy, -0.5)))
        poly = Polygon(vertices, edges=[[0, 1], [1, 2], [2, 0]], color=color, name='polygon_' + str(p_count))
        poly.appear(begin_time=t0, transition_time=0.1 * dt)
        poly.move(direction=[0, 0, 0.52], begin_time=t0, transition_time=dt)
        t0 += dt
        p_count += 1


def geometry2(A, B, C, T, components):
    # Translate A to the origin
    loc_b = B - A
    loc_t = T - A
    loc_c = C - A
    # calculate alpha
    alpha = np.arccos(loc_b.dot(loc_c) / loc_b.length / loc_c.length)
    # go to complex numbers
    b = np.abs(loc_b[components[0]])
    t = np.abs(loc_t[components[0]])
    h = loc_c[components[1]]
    d = b * (2 * t - b) / (h * h + t * t) * (t + 1j * h)
    e = (2 * t - b) / t * b
    c = t + 1j * h
    beta = np.angle(c - b)
    return [d, e]


def get_sphere_location(obj, curve, frm, mapping):
    constraint = ibpy.FOLLOW_PATH_DICTIONARY[(obj, curve)]
    ibpy.set_frame(frm)
    phi = 2 * np.pi * constraint.offset_factor
    return mapping(phi)


def action(pencil, location, direction, ortho,
           size, color, t0, duration, name, case=0):
    dt = duration / 5
    geometry = []

    a = location
    s_a = Sphere(0.25, location=Vector(), color=color)
    s_a.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_a)

    b = location + size * direction
    l_ab = Cylinder.from_start_to_end(start=Vector(), end=b - a, thickness=0.5, color=color)
    l_ab.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move(direction=direction * size, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_ab)

    s_b = Sphere(0.25, location=b - a, color=color)
    s_b.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_b)

    if case == 0:
        pencil.move(direction=-direction * 0.4 * size, begin_time=t0, transition_time=dt / 2)
        t0 += dt / 2
        t = a + 0.6 * size * direction
        s_t = Sphere(0.25, location=t - a, color=color)
        s_t.grow(begin_time=t0, transition_time=dt / 2)
        t0 += dt / 2
        geometry.append(s_t)
    else:
        pencil.move(direction=-direction * 0.1 * size, begin_time=t0, transition_time=dt / 2)
        t0 += dt / 2
        t = a + 0.9 * size * direction
        s_t = Sphere(0.25, location=t - a, color=color)
        s_t.grow(begin_time=t0, transition_time=dt / 2)
        t0 += dt / 2
        geometry.append(s_t)

    return BObject(children=geometry, location=a)


def action2(pencil, location, direction, ortho,
            size, color, t0, duration, name, case=0):
    dt = duration / 5
    geometry = []

    a = location
    s_a = Sphere(0.25, location=Vector(), color=color)
    s_a.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_a)

    b = location + size * direction
    l_ab = Cylinder.from_start_to_end(start=Vector(), end=b - a, thickness=0.5, color=color)
    l_ab.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move(direction=direction * size, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_ab)

    s_b = Sphere(0.25, location=b - a, color=color)
    s_b.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_b)

    pencil.move(direction=-direction * 0.3 * size, begin_time=t0, transition_time=dt / 2)
    t0 += dt / 2
    t = a + 0.7 * size * direction
    s_t = Sphere(0.25, location=t - a, color=color)
    s_t.grow(begin_time=t0, transition_time=dt / 2)
    t0 += dt / 2
    geometry.append(s_t)

    if case == 0:
        c = t - size * ortho
    else:
        c = t + size * ortho

    l_tc = Cylinder.from_start_to_end(start=t - a, end=c - a, color=color, thickness=0.5)
    l_tc.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move_to(target_location=c, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_tc)

    s_c = Sphere(0.25, location=c - a, color=color)
    s_c.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_c)

    return BObject(children=geometry, location=a)


def action3(pencil, location, direction, ortho,
            size, color, t0, duration, name, case=0):
    dt = duration / 5
    geometry = []

    a = location
    s_a = Sphere(0.25, location=Vector(), color=color)
    s_a.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_a)

    b = location + size * direction
    l_ab = Cylinder.from_start_to_end(start=Vector(), end=b - a, thickness=0.5, color=color)
    l_ab.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move(direction=direction * size, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_ab)

    s_b = Sphere(0.25, location=b - a, color=color)
    s_b.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_b)

    pencil.move(direction=-direction * 0.3 * size, begin_time=t0, transition_time=dt / 2)
    t0 += dt / 2
    t = a + 0.7 * size * direction
    s_t = Sphere(0.25, location=t - a, color=color)
    s_t.grow(begin_time=t0, transition_time=dt / 2)
    t0 += dt / 2
    geometry.append(s_t)

    if case == 0:
        c = t - size * ortho
    else:
        c = t + size * ortho

    l_tc = Cylinder.from_start_to_end(start=t - a, end=c - a, color=color, thickness=0.5)
    l_tc.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move_to(target_location=c, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_tc)

    s_c = Sphere(0.25, location=c - a, color=color)
    s_c.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_c)

    return BObject(children=geometry, location=a)


def golden_action(pencil, location, direction, ortho, size, aspectratio, color, t0, duration, name, sign=1):
    dt = duration / 14
    geometry = []

    angle0 = np.angle(direction.x + 1j * direction.y)
    angle1 = np.angle(-ortho.x - 1j * ortho.y)

    a = location
    s_a = Sphere(0.25, location=Vector(), color=color)
    s_a.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_a)

    b = location + size * direction
    l_ab = Cylinder.from_start_to_end(start=Vector(), end=b - a, thickness=0.5, color=color)
    l_ab.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move_to(target_location=b, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_ab)

    s_b = Sphere(0.25, location=b - a, color=color)
    s_b.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_b)

    c = location + size * direction + size * aspectratio * ortho
    l_bc = Cylinder.from_start_to_end(start=b - a, end=c - a, thickness=0.5, color=color)
    l_bc.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move(direction=ortho * size * aspectratio, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_bc)

    s_c = Sphere(0.25, location=c - a, color=color)
    s_c.grow(begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(s_c)

    l_ca = Cylinder.from_start_to_end(start=c - a, end=a - a, thickness=0.5, color=color)
    l_ca.grow(modus='from_start', begin_time=t0, transition_time=dt)
    pencil.move(direction=-direction * size - ortho * size * aspectratio, begin_time=t0, transition_time=dt)
    t0 += dt
    geometry.append(l_ca)

    offset = a + 0.01 * (b - a)
    pencil.move_to(target_location=offset, begin_time=t0, transition_time=0.1 * dt)

    rope = Rope(start=a - a, end=offset - a, length=(b - a).length * 0.975, folding_direction=-ortho.normalized(),
                resolution=10, name=name + '_Rope1', color=color)
    pin = Pin(location=a - a + Vector([0, 0, 10]), colors=[color, 'metal_0.5'], scale=0.5, name=name + '_Pin1')
    pin.appear(begin_time=t0)
    geometry.append(pin)
    rope.attach_to(s_a, pencil)

    pin.move(direction=[0, 0, -10], begin_time=t0)
    rope.appear(begin_time=t0)
    rope.set_dynamic(begin_time=t0, transition_time=4 * dt)
    geometry.append(rope)
    t0 += dt

    pencil.move_to(target_location=b, begin_time=t0, transition_time=dt)
    t0 += dt

    angle = sign * np.arccos((b - a).dot((c - a)) / (b - a).length / (c - a).length)
    start = angle0
    end = angle0 + angle

    arc1 = CircleArc(center=a - a, radius=(b - a).length, start_angle=start, end_angle=end,
                     color=color, thickness=0.5, name=name + "_Arc1")
    for i in range(0, 11):
        ddt = dt / 10
        da = angle / 10
        alpha = angle0 + i * da
        if i > 0:
            arc1.grow(begin_time=t0 + i * ddt, transition_time=ddt, start_factor=(i - 1) * da / angle,
                      end_factor=i * da / angle)
        if sign > 0:
            pencil.move_to(
                target_location=a - direction * (b - a).length * np.sin(alpha) + sign * ortho * (b - a).length * np.cos(
                    alpha), begin_time=t0 + i * ddt, transition_time=ddt)
        else:
            pencil.move_to(
                target_location=a - direction * (b - a).length * np.cos(alpha) + ortho * (b - a).length * np.sin(alpha),
                begin_time=t0 + i * ddt, transition_time=ddt)

    geometry.append(arc1)
    t0 += dt

    d = a + (c - a).normalized() * (b - a).length
    s_d = Sphere(0.25, location=d - a, color=color)
    s_d.grow(begin_time=t0, transition_time=dt)
    geometry.append(s_d)

    rope.disappear(begin_time=t0)
    pin.disappear(begin_time=t0)
    t0 += dt

    pencil.move_to(target_location=c, begin_time=t0)
    rope2 = Rope(start=c - a, end=c - a, length=(d - c).length * 0.975, folding_direction=-ortho.normalized(),
                 resolution=10, name=name + '_Rope2', color=color)
    pin2 = Pin(location=c - a + Vector([0, 0, 10]), colors=[color, 'metal_0.5'], scale=0.5, name=name + '_Pin2')
    pin2.appear(begin_time=t0)
    geometry.append(pin2)
    geometry.append(rope2)
    rope2.attach_to(s_c, None)

    pin2.move(direction=[0, 0, -10], begin_time=t0)
    rope2.appear(begin_time=t0 + 0.5 * dt)
    rope2.set_dynamic(begin_time=t0 + 0.5 * dt, transition_time=4 * dt)
    t0 += dt

    pencil.move_to(target_location=d, begin_time=t0)
    rope2.hooks[1].move_to(target_location=d - a, begin_time=t0)
    t0 += dt

    angle = sign * np.arccos((d - c).dot((b - c)) / (b - c).length / (d - c).length)
    start = angle1 - angle
    end = angle1
    arc2 = CircleArc(center=c - a, radius=(d - c).length, start_angle=start, end_angle=end,
                     color=color, thickness=0.5, name=name + '_Arc2')

    r = (d - c).length
    for i in range(0, 11):
        ddt = dt / 10
        da = angle / 10
        alpha = start + i * da
        if i > 0:
            arc2.grow(begin_time=t0 + i * ddt, transition_time=ddt, start_factor=(i - 1) * da / angle,
                      end_factor=i * da / angle)
        if sign > 0:
            loc = c - direction * r * np.sin(alpha) + ortho * r * np.cos(alpha)
        else:
            loc = c - direction * r * np.cos(alpha) + ortho * r * np.sin(alpha)
        pencil.move_to(
            target_location=loc,
            begin_time=t0 + i * ddt, transition_time=ddt)
        rope2.hooks[1].move_to(target_location=loc - a, begin_time=t0 + i * ddt, transition_time=ddt)

    geometry.append(arc2)
    t0 += dt

    e = c + (b - c).normalized() * (d - c).length
    s_e = Sphere(0.25, location=e - a, color=color)
    geometry.append(s_e)
    s_e.grow(begin_time=t0)
    rope2.disappear(begin_time=t0)
    pin2.disappear(begin_time=t0)
    t0 += dt

    return BObject(children=geometry, location=a)


def geometry(A, B, C, T):
    # Translate A to the origin
    loc_b = B - A
    loc_t = T - A
    loc_c = C - A
    # calculate alpha
    alpha = np.arccos(loc_b.dot(loc_c) / loc_b.length / loc_c.length)
    # go to complex numbers
    b = loc_b[0]
    t = loc_t[0]
    h = loc_c[2]
    d = b * (2 * t - b) / (h * h + t * t) * (t + 1j * h)
    e = (2 * t - b) / t * b
    c = t + 1j * h
    beta = np.angle(c - b)
    return [Vector((np.real(d), 0, np.imag(d))) + A, alpha, beta, Vector((e, 0, 0)) + A]


class BundesWettbewerbGeometry(Scene):
    def __init__(self):
        self.construction_counter = 0
        self.sheet = None
        self.people = None
        self.sub_scenes = OrderedDict([
            ('thumbnail', {'duration': 0}),
            ('intro_animation', {'duration': 0}),
            ('some2', {'duration': 0}),
            ('intro', {'duration': 0}),
            ('title', {'duration': 0}),
            ('question', {'duration': 24}),
            ('addition_formal', {'duration': 10}),
            ('addition', {'duration': 10}),
            ('scaling', {'duration': 10}),
            ('multiplication_formal', {'duration': 22}),
            ('complex_plane', {'duration': 20}),
            ('complex_plane2', {'duration': 20}),
            ('complex_plane_walking', {'duration': 20}),
            ('norm_product', {'duration': 20}),
            ('lines', {'duration': 20}),
            ('svg', {'duration': 20}),
            ('i2m_one', {'duration': 20}),
            ('debug2', {'duration': 20}),
            ('basics', {'duration': 20}),
            ('basics2', {'duration': 20}),
            ('notation', {'duration': 20}),
            ('wlog', {'duration': 20}),
            ('wlog2', {'duration': 20}),
            ('wlog3', {'duration': 20}),
            ('wlog4', {'duration': 20}),
            ('wlog4b', {'duration': 20}),
            ('multiplication', {'duration': 20}),
            ('rotation', {'duration': 20}),
            ('solution1', {'duration': 20}),
            ('solution1b', {'duration': 20}),
            ('solution1c', {'duration': 20}),
            ('solution1d', {'duration': 20}),
            ('solution2', {'duration': 20}),
            ('solution2b', {'duration': 20}),
            ('solution2c', {'duration': 20}),
            ('solution2d', {'duration': 20}),
            ('further_reading', {'duration': 20}),
        ])
        # in the super constructor the timing is set for all scenes
        super().__init__(light_energy=2)

    def intern_geometry(self,A, B, C, T):
        # Translate A to the origin
        loc_b = B - A
        loc_t = T - A
        loc_c = C - A
        # calculate alpha
        alpha = np.arccos(loc_b.dot(loc_c) / loc_b.length / loc_c.length)
        # go to complex numbers
        b = loc_b[0]
        t = loc_t[0]
        h = loc_c[2]
        d = b * (2 * t - b) / (h * h + t * t) * (t + 1j * h)
        e = (2 * t - b) / t * b
        c = t + 1j * h
        beta = np.angle(c - b)
        return [Vector((np.real(d), 0, np.imag(d))) + A, alpha, beta, Vector((e, 0, 0)) + A]

    def title(self):
        cues = self.sub_scenes['title']
        t0 = 0  # cues['start']

        working_title = SimpleTexBObject(r"\text{Solving Olympiad Level Geometry Problems }",
                                         color='metal_0.8', aligned="center", scale=3, location=[0, 0, 3], bevel=0.75)
        working_title.write(begin_time=t0, transition_time=5)
        t0 += 5

        working_title = SimpleTexBObject(r"\text{with}", color='text',
                                         aligned="center", scale=3)
        working_title.write(begin_time=t0, transition_time=1)
        t0 += 1

        title = SimpleTexBObject(r"\text{Complex Numbers}", color='marble', scale=5, name='title',
                                 rotation_euler=[-np.pi / 2, 0, 0],location=[0, 0, -3], aligned="center", bevel=0.8)
        title.write(begin_time=t0, transition_time=2.5)

        plane = Plane(color='mirror', location=[0, 0, 0], scale=[7.756, 6, 1])
        plane.appear(begin_time=t0, transition_time=0.5)
        plane.move(direction=[0, 0, -4], begin_time=t0, transition_time=0)
        t0 += 3

        title.rotate(rotation_euler=[np.pi/2, 0, 0], begin_time=t0, transition_time=5)
        t0 += 5

    def intro(self):
        cues = self.sub_scenes['intro']
        t0 = 0  # cues['start']
        angle = np.pi / 4

        ibpy.set_camera_location(location=[0, 0, 0])
        circle_a = Curve(lambda phi: [0, -20 * np.cos(phi), 20 * np.sin(phi)], domain=[0, angle],
                         thickness=0, name='circle_a')
        circle_b = Curve(lambda phi: [11.5, 6 * np.sin(phi), 6 * np.cos(phi)], domain=[0, angle],
                         thickness=0, name='circle_b')
        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)
        ibpy.set_camera_follow(target=circle_a)

        logo = SimpleTexBObject("bwm_logo", color=['custom1', 'custom2', 'custom3', 'custom4', 'custom5', 'text'],
                                aligned='right', scale=0.15, rotation_euler=[np.pi / 2, 0, 0], name='Logo',
                                shadow=False,emission=0)
        set = [0, 1, 2, 3, 4, 5, 11, 15, 19, 22, 26, 31, 35, 37, 39, 41, 42, 43, 44, 45, 46, 6, 12, 17, 20, 23, 28, 33,
               36, 38, 40, 9, 7, 8, 10, 13, 14, 16, 18, 21, 24, 25, 27, 29, 30, 32, 34]
        logo.write(letter_set=set,begin_time=0, transition_time=2)

        logo_translation = [
            SimpleTexBObject(r"\text{$^\star${\bf FEDERAL COMPETITION}}", color='text', aligned='left',
                             name='Translation1',
                             scale=1.2, shadow=False, location=[6.259, 0, 4.81]),
            SimpleTexBObject(r"\text{{\bf MATHEMATICS}}", color='text', aligned='left', name='Translation2',
                             scale=1.2, shadow=False, location=[6.45, 0, 4.3]),
            SimpleTexBObject(r"\text{\bf Education \& Talent}", color='text', aligned='left', name='Translation3',
                             scale=0.8, shadow=False, location=[6.45, 0, 3.87])
        ]

        for trans in logo_translation:
            trans.write(begin_time=2, transition_time=0)

        ibpy.set_follow(logo, circle_b)
        t0 += 2
        logo.scale(initial_scale=[1, 1, 1], final_scale=[1, 1, 5], begin_time=t0, transition_time=0.5)

        t0 += 1
        self.sheet = Display(number_of_lines=40, rotation_euler=[np.pi / 2, 0, np.pi], scales=[4, 6], color='gray_7',
                             flat=True, location=[-6, 0, -0.1])
        self.sheet.appear(begin_time=t0, transition_time=1)

        title = SimpleTexBObject("bwm_logo",
                                 color=['custom1', 'custom2', 'custom3', 'custom4', 'custom5', 'background'],
                                 aligned='right', location=[10, 0, 5], name='text_logo',emission=0);
        title2 = SimpleTexBObject("bwm_logo",
                                  color=['custom1', 'custom2', 'custom3', 'custom4', 'custom5', 'background'],
                                  aligned='right', location=[10, 0, 5], name='text_logo_2',emission=0);
        self.sheet.set_title(title, shift=[-0.5, 4.5], scale=0.1)
        self.sheet.set_title_back(title2, shift=[-0.5, 4.5], scale=0.1)

        t0 += 1.5
        title.write(begin_time=t0, transition_time=1)
        title2.write(letter_set=set,begin_time=t0, transition_time=1)

        second_round = SimpleTexBObject(r"\text{Round 2: 2018}", color='important', aligned='left',emission=0)
        first_round = SimpleTexBObject(r"\text{Round 1: 2018}", color='important', aligned='left',emission=0)
        self.sheet.add_text_in_back(first_round, scale=1.4, indent=1, line=3)
        self.sheet.add_text_in(second_round, scale=1.4, indent=1, line=3)
        second_round.write(begin_time=t0, transition_time=0.5)
        first_round.write(begin_time=t0, transition_time=0.5)

        t0 += 1
        problems = []
        problems_back = []
        for i in range(4):
            problems.append(SimpleTexBObject(r"\text{Problem }\," + str(i + 1), color='important', aligned='left',emission=0))
            self.sheet.add_text_in(problems[-1], scale=1, indent=1.5, line=i * 8 + 5)
            problems[-1].write(begin_time=t0, transition_time=0.5)
            problems_back.append(SimpleTexBObject(r"\text{Problem }\," + str(i + 1), color='important', aligned='left',emission=0))
            self.sheet.add_text_in_back(problems_back[-1], scale=1, indent=1.5, line=i * 8 + 5)
            problems_back[-1].write(begin_time=t0, transition_time=0.5)
            t0 += 1

        t0 += 0.5
        for i in range(4):
            if i == 3:
                correct = False
            else:
                correct = True
            t0 = self.mark_back(i + 1, t0, correct)

        # change perspective
        for trans in logo_translation:
            trans.disappear(begin_time=t0)

        t0 += 0.5
        self.sheet.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        self.sheet.rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=1)
        ibpy.camera_follow(circle_a, initial_value=0, final_value=1, begin_time=t0, transition_time=1)
        ibpy.follow(logo, circle_b, initial_value=0, final_value=1, begin_time=t0, transition_time=1)
        self.sheet.move(direction=[6, 0, 0], begin_time=t0, transition_time=1)

        floor = Floor(u=[-20, 20], v=[-20, 20], location=[0, 0, -0.01])
        t0 += 1
        floor.appear(begin_time=t0, transition_time=1)

        self.number_of_people = 150

        t0 += 1
        dt = 1
        self.populate(t0, dt)
        dt = 6
        t0 += dt
        play_problem1(t0, dt)
        t0 += dt + 0.5
        self.mark(1, t0)
        t0 += 1.5
        self.play_problem2(t0, dt)
        t0 += dt + 0.5
        self.mark(2, t0)
        t0 += 1.5
        self.play_problem3(t0, dt)
        t0 += dt + 0.5
        self.mark(3, t0)
        t0 += 1.5
        play_problem4(t0, dt)
        t0 += dt + 0.5
        self.mark(4, t0)
        t0 += 1.5

        self.ausgang = [-14, 10, 0.36]
        ausgang2 = [-14,10,0]
        t0 += 0.5

        wall_with_door = WallWithDoor(colors=['custom2', 'custom1'], location=ausgang2, rotation_euler=[0, 0, 0])
        wall_with_door.appear(begin_time=t0, transition_time=1)
        third_round = SimpleTexBObject(r"\text{Round 3}", color='important', aligned='center',
                                       location=[-11.5, 9.8, 1.8],
                                       rotation_euler=[np.pi / 2, 0, 0], scale=2)
        third_round.write(begin_time=t0 + 0.3, transition_time=0.5)
        t0 += 0.5
        wall_with_door.open_door(begin_time=t0, transition_time=0.5)
        t0 += 0.5
        t0 = self.leave(t0, 5)
        print("End at ", t0)

    def leave(self, t0, duration):
        dt = duration / len(self.people)
        for p in self.people:
            p.move_to(target_location=self.ausgang, begin_time=t0, transition_time=10 * dt)
            p.move(direction=[0, 1, 0], begin_time=t0 +10*dt, transition_time= dt)
            p.disappear(begin_time=t0 + 11*dt, transition_time=dt)
            t0 += dt
        t0 += 0.5
        return t0

    def mark(self, number, t0):
        check = SimpleTexBObject(r"\text{\textcircled{$\checkmark$}}", color='important')
        self.sheet.add_text_in(check, line=1 + number * 8, indent=9, scale=3)
        check.write(begin_time=t0, transition_time=1)
        dt = 1 / 30
        for i in range(int(self.number_of_people / 5)):
            person = self.people[0]
            person.disappear(begin_time=t0, transition_time=dt)
            person.hide(begin_time=t0 + dt)
            t0 += dt
            self.people.remove(person)

    def mark_back(self, number, t0, value):
        if value:
            check = SimpleTexBObject(r"\text{\textcircled{$\checkmark$}}", color='important')
            scale = 3
            indent = 9
        else:
            check = SimpleTexBObject(r"\varnothing", color='important')
            scale = 4.5
            indent = 8.9
        self.sheet.add_text_in_back(check, line=1 + number * 8, indent=indent, scale=scale)
        check.write(begin_time=t0, transition_time=1)
        t0 += 1.5
        return t0

    def check_vicinity(self, location):
        for loc in self.locations:
            if (loc - location).length < 0.5:
                return False
        return True

    def populate(self, t0, duration):
        self.locations = []
        count = 0
        colors = ['custom1', 'custom2', 'custom3', 'custom4', 'custom5']
        dt = duration / 150
        person_scale = [0.25] * 3
        self.people = []
        while count < self.number_of_people:
            x = np.random.rand() * 18 - 9
            y = np.random.rand() * 16.5 - 7
            location = Vector([x, y, 0])
            c = np.random.randint(0, 5)
            if y < -6.1 or y > 6.1:
                if self.check_vicinity(location):
                    person = BObject.from_file('Person', color=colors[c], location=[x, y, 0.36],
                                               name='Person_' + str(count),
                                               scale=person_scale)
                    self.locations.append(location)
            elif x < -4.1 or x > 4.1:
                if self.check_vicinity(location):
                    person = BObject.from_file('Person', color=colors[c], location=[x, y, 0.36],
                                               name='Person_' + str(count),
                                               scale=person_scale)
                    self.locations.append(location)
            else:
                person = None
            if person:
                self.people.append(person)
                count += 1
                person.un_hide(begin_time=t0)
                person.appear(begin_time=t0, transition_time=dt)
                t0 += dt

    def play_problem2(self, t0, duration):
        functional = SimpleTexBObject(r"f\left(1-f(x)\right)=x",
                                      color='joker', thickness=5, shading='darker', emission=0)
        self.sheet.add_text_in(functional, line=16, indent=1, scale=1.4)
        functional.write(begin_time=t0, transition_time=duration)

        coords = CoordinateSystem(dim=2, lengths=[3, 3], domains=[[-6, 6], [-6, 6]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"", r""],
                                  all_tic_labels=[np.arange(-6, 6.1, 2), np.arange(-6, 6.1, 2)],
                                  colors=['background', 'background'], label_colors=['background', 'background'],
                                  label_digits=[0, 0], label_units=['', ''],
                                  axis_label_size='medium',
                                  tic_label_size='small', name='Solution', rotation_euler=[-np.pi / 2, 0, 0],
                                  location_of_origin=[1.5, 0, 0])
        coords.appear(begin_time=t0, transition_time=duration / 2)
        graph = SVGBObject('graph', color='joker', name='Graph', aligned='center', rotation_euler=[np.pi / 2, 0, 0],
                           location=coords.coords2location([0, 0.5]), bevel=7, scale=0.27)
        coords.add_object(graph)
        graph.appear(begin_time=t0 + duration / 2, transition_time=duration / 2)

    def play_problem3(self, t0, duration):
        dt = duration * 0.1
        line_size = 0.1
        point_size = 0.5

        origin = Vector((-10, 0, -5))

        a_pos = origin
        b_pos = origin + 10 * RIGHT
        t_pos = origin + 7 * RIGHT
        c_pos = origin + 7 * RIGHT + 6 * UP

        s_a = Sphere(point_size, location=a_pos, color='drawing', name='A', smooth=2)
        s_b = Sphere(point_size, location=b_pos, color='drawing', name='B', smooth=2)
        s_c = Sphere(point_size, location=c_pos, color='joker', name='C', smooth=2)
        s_t = Sphere(point_size, location=t_pos, color='example', name='T', smooth=2)
        l_ab = Cylinder.from_start_to_end(start=origin, end=b_pos, color='drawing', radius=line_size)
        l_ac = Cylinder.from_start_to_end(start=origin, end=c_pos, color='drawing', radius=line_size)

        s_a.appear(begin_time=t0, transition_time=dt)
        s_a.write_name_as_label(modus='down', begin_time=t0, transition_time=dt)
        s_b.appear(begin_time=t0, transition_time=dt)
        s_b.write_name_as_label(modus='down', begin_time=t0, transition_time=dt)
        l_ab.grow(begin_time=t0, transition_time=dt, modus='from_start')

        ab = b_pos - a_pos
        s_t.appear(begin_time=t0, transition_time=dt)
        s_t.write_name_as_label(modus='down_right', begin_time=t0, transition_time=dt)
        s_c.appear(begin_time=t0, transition_time=dt)
        s_c.write_name_as_label(modus='up_right', begin_time=t0, transition_time=dt)

        perp = Cylinder.from_start_to_end(start=t_pos + DOWN, end=c_pos + UP, color='example', radius=line_size)
        perp.grow(begin_time=t0, transition_time=dt)
        [d_pos, alpha, beta, e_pos] = self.intern_geometry(a_pos,b_pos,c_pos,t_pos)
        s_d = Sphere(point_size, location=d_pos, color='important', name='D', smooth=2)
        s_e = Sphere(point_size, location=e_pos, color='important', name='E', smooth=2)
        s_d.appear(begin_time=t0, transition_time=dt)
        s_d.write_name_as_label(modus='up', begin_time=t0, transition_time=dt)
        l_ac.grow(begin_time=t0, transition_time=dt, modus='from_start')
        bad = CircleArc(center=a_pos, radius=2.5, start_angle=0, end_angle=np.pi / 2, color='important', name='Arc_bad',
                        mode='XZ', thickness=1)
        bad.appear(begin_time=t0, transition_time=dt)
        bad.extend_to(2 * alpha / np.pi, begin_time=t0, transition_time=0)
        bad.write_name_as_label(modus='center', begin_time=t0, transition_time=dt, name=r"\angle BAD")

        aux_line1 = Cylinder.from_start_to_end(start=b_pos, end=c_pos, color='gray_5', radius=line_size)
        aux_line2 = Cylinder.from_start_to_end(start=b_pos, end=d_pos, color='gray_5', radius=line_size)
        aux_line1.grow(begin_time=t0, transition_time=dt, modus='from_start')
        aux_line2.grow(begin_time=t0, transition_time=dt, modus='from_start')

        cbd = CircleArc(center=b_pos, radius=2.5, start_angle=0, end_angle=np.pi, color='important',
                        name='Arc_cbd', mode='XZ', thickness=1)
        cbd.appear(begin_time=t0, transition_time=dt)
        cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t0, transition_time=0)
        cbd.write_name_as_label(modus='center', begin_time=t0, transition_time=dt, name=r"\angle CBD")
        l_de = Cylinder.from_start_to_end(start=e_pos, end=d_pos, color='example', radius=line_size)
        l_de.grow(begin_time=t0, transition_time=dt, modus='from_start')
        s_e.appear(begin_time=t0, transition_time=dt)
        s_e.write_name_as_label(modus='down', begin_time=t0, transition_time=dt)

        t0 += dt

        duration = 0.4 * duration
        direction = Vector([0, 0, -2])

        s_c.move(direction=direction, begin_time=t0, transition_time=duration)
        steps = 10
        dt = duration / steps
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = self.intern_geometry(a_pos, b_pos, c_new, t_pos)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bad.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            aux_line1.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            aux_line2.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += duration + 0.5

        direction = Vector([0, 0, 2])

        s_c.move(direction=direction, begin_time=t0, transition_time=duration)
        steps = 10
        dt = duration / steps
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = self.intern_geometry(a_pos, b_pos, c_new, t_pos)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bad.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            aux_line1.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            aux_line2.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += duration + 0.5

        geometry = BObject(
            children=[l_ab, l_ac, l_de, s_a, s_b, s_c, s_d, s_e, s_t, aux_line1, aux_line2, bad, cbd, l_de, perp],
            rotation_euler=[-np.pi / 2, 0, 0], scale=0.05, location=[0, -0.2, 0.025], name="Geometry")
        geometry.appear(begin_time=0, transition_time=0)
        geometry.ref_obj.parent = self.sheet.ref_obj

    def thumbnail(self):
        cues = self.sub_scenes['thumbnail']
        t0 = 0  # cues['start']

        [a, b, c, t] = [-1, 1, 0.5 + 2j, 0.5]
        dic = self.construction(t0, "Thumbnail",
                                locs=[a, b, c, t],
                                durations=[1,1,1,1], text=False)

        ibpy.set_hdri_background('autoshop_01_4k')
        ibpy.set_hdri_strength(1, begin_time=0)

        line = SimpleTexBObject(r"i^2=-1", aligned='center', text_size='huge', color='gold',
                             thickness=5, bevel=3, name='Treasure', emission=1,location=[-5.3,0,3])

        line.write(begin_time=t0)

        arrow = PArrow(start=[-5,0,0],end=[-4,0,0],color='text')
        arrow.grow(begin_time=t0,transition_time=0)

        sphere_volume = Sphere(10, color='scatter_volume', name='ScatteringSphere')
        sphere_volume.appear(begin_time=t0, transition_time=0)
        ibpy.set_volume_scatter(sphere_volume, 0.0025, begin_time=t0)
        ibpy.change_volume_scatter(sphere_volume, 0, begin_time=t0, transition_time=1)

        ibpy.set_volume_absorption(sphere_volume, 0.0, begin_time=t0)
        ibpy.change_volume_absorption(sphere_volume, 0.01, begin_time=t0 + 0.1, transition_time=1)


    def intro_animation(self):
        cues = self.sub_scenes['intro_animation']
        t0 = 0  # cues['start']

        width = 20
        height = 10

        logo = LogoPreImage(rotation_euler=[0, 0, np.pi / 2], scale=10, dim=50)
        logo.appear(begin_time=t0, transition_time=7)

        logo.rotate(rotation_euler=[np.pi / 4, 0, np.pi], begin_time=t0, transition_time=3.5)
        logo.move(direction=[0, 4, 1.2], begin_time=t0, transition_time=2)
        logo.transform(transformation=lambda v: riemann_r(v[0], v[1], 0.5), begin_time=t0, transition_time=3.5)
        # logo.scale(initial_scale=10,final_scale=12,begin_time=t0+5,transition_time=10)

        nc = SimpleTexBObject(r"\text{NumberCruncher}", scale=3, aligned="center", name="nc", location=[0, 0, -4])
        nc.write(begin_time=t0 + 3.5, transition_time=1)

        t0 += 5
        logo.move(direction=[width / 4, 0, 0], begin_time=t0, transition_time=1)
        nc.move(direction=[width / 4, 0, 0], begin_time=t0, transition_time=1)

        t0 += 1.5
        puzzle = SliderPuzzle(size=8, location=[-8, 0, -height / 4 + 0.5])
        author = SimpleTexBObject(r"\text{Alexandru \scshape{Duca}}", scale=3, aligned="center", name="author",
                                  location=[-5.3, 0, -4])

        puzzle.appear(begin_time=t0, transition_time=1)
        puzzle.solve(begin_time=t0 + 1.5, transition_time=5)
        author.write(begin_time=t0 + 2, transition_time=1)

        t0+=7.25
        coop = SimpleTexBObject(r"\text{In cooperation with}",scale=1.5,aligned='right',name='coop',location=[-1.25,0,-6])
        coop.write(begin_time=t0,transition_time=2)
        logo = SimpleTexBObject("bwm_logo", color=['custom1', 'custom2', 'custom3', 'custom4', 'custom5', 'text'],
                                aligned='left', scale=0.15, rotation_euler=[np.pi / 2, 0, 0],location=[-1,0,-6], name='Logo',
                                shadow=False,emisssion=0)
        set = [0,1,2,3,4,5,11,15,19,22,26,31,35,37,39,41,42,43,44,45,46,6,12,17,20,23,28,33,36,38,40,9,7,8,10,13,14,16,18,21,24,25,27,29,30,32,34]
        logo.write(letter_set=set,begin_time=t0+3, transition_time=2)
        t0+=6.25

        print(t0)

    def some2(self):
        cues = self.sub_scenes['some2']
        t0 = 0  # cues['start']

        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)
        text = SimpleTexBObject(r"\text{The colors in this video have been adjusted to be distinguishable for people suffering from }",aligned='center',color='text',location=[0,0,6],scale=5/3)
        text2 = SimpleTexBObject(r"\text{Deuteranopia, Protanopia and Tritanopia.}",aligned='center',color='text',location=[0,0,5],scale=5/3)
        text.write(begin_time=t0,transition_time=2)
        text2.write(begin_time=t0+1,transition_time=1)

        t0+=2
        shift = 2 * DOWN
        title = SimpleTexBObject(r"\text{SoME2}", color=['text'], scale=5, extrude=5, aligned="center",
                                 location=shift,bevel=0.05,thickness=1)
        title.write(letter_range=[0, 4], begin_time=t0, transition_time=1)
        title.grow_letter(index=4, final_scale=0.5, begin_time=t0 + 1, transition_time=0.25)
        subtitle = SimpleTexBObject(r"\text{Summer of Math Exposition \#2}", color=["text"],bevel=0,thickness=0,
                                    scale=5 / 3, aligned="center", location=shift + Vector([0, 0, -1.5]))
        t0 += 1
        title.rotate(rotation_euler=[np.pi/2,0,2*np.pi],begin_time=t0,transition_time=1.5)
        subtitle.write(begin_time=t0, transition_time=1)
        plane = Plane(color='mirror', location=[0, -1.73, -3.8468], scale=[4.2, 4, 1])
        # plane = Plane(color='mirror', location=[0, -1.73, 0.02], scale=[4.2, 1.4, 1])
        plane.appear(begin_time=t0, transition_time=0.5)
        plane.appear(begin_time=t0, transition_time=0.5)

    def question(self):
        cues = self.sub_scenes['question']
        t0 = 0  # cues['start']

        display = Display(scales=[5, 6], location=[6, 0, 0], number_of_lines=18)
        title = SimpleTexBObject(r"\text{Problem 3}", color='important', aligned='center')
        display.set_title(title)
        title.write(begin_time=t0, transition_time=2)
        t0 += 2

        colors = [
            flatten([['text'] * 3, ['drawing'] * 3, ['text']]),
            flatten([['text'] * 6, ['example'], ['text']]),
            flatten([['text'] * 8, ['drawing'], ['text'] * 6, ['drawing']]),
            flatten([['text'] * 3, ['joker'], ['text'] * 13, ['example']]),
            flatten([['text'] * 6, ['example'] * 13, ['text'] * 2, ['drawing'] * 3]),
            flatten([['text'] * 14, ['example']]),
            flatten([['text']]),
            flatten([['text'] * 15, ['important'], ['text'] * 2, ['drawing'] * 2]),
            flatten([['text']]),
            flatten([['important'] * 4, ['text'] * 3, ['important'] * 4, ['text']]),
            flatten([['text'] * 14, ['example'] * 4, ['text'] * 7, ['important']]),
            flatten([['text'] * 3, ['example'] * 13, ['text'] * 2, ['drawing']]),
            flatten([['text'] * 10, ['drawing'] * 3, ['text'] * 8, ['important'], ['text']]),
            flatten([['text'] * 26, ['joker']]),
        ]

        lines = [
            SimpleTexBObject(r"\text{Let $\overline{AB}$ be a line segment}", aligned='left', color=colors[0]),
            SimpleTexBObject(r"\text{and let $T$ be a point on it}", aligned='left', color=colors[1]),
            SimpleTexBObject(r"\text{closer to $B$ than to $A$.}", aligned='left', color=colors[2]),
            SimpleTexBObject(r"\text{Let $C$ be a point on the line}", aligned='left', color=colors[3]),
            SimpleTexBObject(r"\text{that is perpendicular to $\overline{AB}$}", aligned='left', color=colors[4]),
            SimpleTexBObject(r"\text{and goes through $T$.}", aligned='left', color=colors[5]),
            SimpleTexBObject(r"\text{1.) Show that there is }", aligned='left', color=colors[6]),
            SimpleTexBObject(r"\text{exactly one point $D$ on $\overline{AC}$}", aligned='left', color=colors[7]),
            SimpleTexBObject(r"\text{such that the angles}", aligned='left', color=colors[8]),
            SimpleTexBObject(r"\text{$\angle CBD$ and $\angle BAC$ are the same.}", aligned='left', color=colors[9]),
            SimpleTexBObject(r"\text{2.) Show that the line through $D$}", aligned='left', color=colors[10]),
            SimpleTexBObject(r"\text{and perpendicular to $\overline{AC}$}", aligned='left', color=colors[11]),
            SimpleTexBObject(r"\text{intersects $\overline{AB}$ in a point $E$ that}", aligned='left',
                             color=colors[12]),
            SimpleTexBObject(r"\text{does not depend on the choice of $C$.}", aligned='left', color=colors[13]),
        ]

        duration = 40
        dt = duration / len(lines) / 2
        dt2 = 2 * dt
        for i, line in enumerate(lines):
            if i in {7, 8, 9, 11, 12, 13, 14}:
                indent = 1
            else:
                indent = 0.5
            if i > 5:
                sep = 2
            else:
                sep = 1
            display.add_text_in(line, line=i + sep, indent=indent)
            line.write(begin_time=t0, transition_time=dt)
            t0 += 2 * dt

        t0 -= duration
        line_size = 0.05
        point_size = 0.25

        origin = Vector((-10, 0, -5))

        a_pos = origin
        b_pos = origin + 10 * RIGHT
        t_pos = origin + 5.5 * RIGHT
        c_pos = origin + 7 * RIGHT + 10 * UP

        s_a = Sphere(point_size, location=a_pos, color='drawing', name='A', smooth=2)
        s_b = Sphere(point_size, location=b_pos, color='drawing', name='B', smooth=2)
        s_c = Sphere(point_size, location=c_pos, color='joker', name='C', smooth=2)
        s_t = Sphere(point_size, location=t_pos, color='example', name='T', smooth=2)
        l_ab = Cylinder.from_start_to_end(start=origin, end=b_pos, color='drawing', radius=line_size)
        l_ac = Cylinder.from_start_to_end(start=origin, end=c_pos, color='drawing', radius=line_size)

        s_a.appear(begin_time=t0, transition_time=dt)
        s_a.write_name_as_label(modus='down', begin_time=t0, transition_time=dt)
        s_b.appear(begin_time=t0, transition_time=dt)
        s_b.write_name_as_label(modus='down', begin_time=t0, transition_time=dt)
        l_ab.grow(begin_time=t0, transition_time=dt, modus='from_start')
        t0 += dt2

        ab = b_pos - a_pos
        s_t.appear(begin_time=t0, transition_time=dt)
        s_t.write_name_as_label(modus='down_right', begin_time=t0, transition_time=dt)

        # giggle t-sphere
        t0 += dt
        s_t.move(direction=0.45 * ab, begin_time=t0, transition_time=dt2)
        t0 += dt2
        s_t.move(direction=-0.3 * ab, begin_time=t0, transition_time=dt2)
        t0 += dt2

        s_c.appear(begin_time=t0, transition_time=dt)
        s_c.write_name_as_label(modus='up_right', begin_time=t0, transition_time=dt)
        t0 += dt

        t_pos = origin + 7 * RIGHT
        perp = Cylinder.from_start_to_end(start=t_pos + DOWN, end=c_pos + UP, color='example', radius=line_size)
        perp.grow(begin_time=t0, transition_time=2 * dt)
        ra = RightAngle(location=t_pos,radius=0.65,thickness=0.4,mode='XZ',color='example',name='RA_C')
        ra.appear(begin_time=t0,transition_time=2*dt)

        t0 += (6 * dt)

        [d_pos, alpha, beta, e_pos] = geometry(a_pos, b_pos, c_pos, t_pos)
        s_d = Sphere(point_size, location=d_pos, color='important', name='D', smooth=2)
        s_e = Sphere(point_size, location=e_pos, color='important', name='E', smooth=2)

        s_d.appear(begin_time=t0, transition_time=dt)
        s_d.write_name_as_label(modus='up', begin_time=t0, transition_time=dt)

        t0 += dt
        l_ac.grow(begin_time=t0, transition_time=dt, modus='from_start')

        t0 += 2 * dt
        bad = CircleArc(center=a_pos, radius=2.5, start_angle=0, end_angle=np.pi / 2, color='important', name='Arc_bad',
                        mode='XZ', thickness=0.5)
        bad.appear(begin_time=t0, transition_time=dt)
        bad.extend_to(2 * alpha / np.pi, begin_time=t0, transition_time=0)
        bad.write_name_as_label(modus='center', begin_time=t0, transition_time=dt, name=r"\angle BAC")

        aux_line1 = Cylinder.from_start_to_end(start=b_pos, end=c_pos, color='text', radius=line_size / 2)
        aux_line2 = Cylinder.from_start_to_end(start=b_pos, end=d_pos, color='text', radius=line_size / 2)

        aux_line1.grow(begin_time=t0, transition_time=dt, modus='from_start')
        aux_line2.grow(begin_time=t0, transition_time=dt, modus='from_start')

        cbd = CircleArc(center=b_pos, radius=2.5, start_angle=0, end_angle=np.pi, color='important',
                        name='Arc_cbd', mode='XZ', thickness=0.5)
        cbd.appear(begin_time=t0, transition_time=dt)
        cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t0, transition_time=0)
        cbd.write_name_as_label(modus='center', begin_time=t0, transition_time=dt, name=r"\angle CBD")
        t0 += 3 * dt

        l_de = Cylinder.from_start_to_end(start=e_pos, end=d_pos, color='example', radius=line_size)
        l_de.grow(begin_time=t0, transition_time=dt, modus='from_start')

        t0 += dt
        ra2 = RightAngle(location=d_pos, rotation_euler=[0,np.pi - alpha, 0], radius=0.65, thickness=0.4,
                         mode='XZ', color='example',name='RA_D')
        ra2.appear(begin_time=t0, transition_time=dt)
        s_e.appear(begin_time=t0, transition_time=dt)
        s_e.write_name_as_label(modus='down', begin_time=t0, transition_time=dt)

        t0 += dt

        duration = 3
        direction = Vector([0, 0, -8])

        s_c.move(direction=direction, begin_time=t0, transition_time=duration)
        steps = 10
        dt = duration / steps
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = geometry(a_pos, b_pos, c_new, t_pos)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bad.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            aux_line1.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            aux_line2.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            ra2.rotate(rotation_euler=[0,np.pi-alpha,0],begin_time=t,transition_time=dt)
            ra2.move_to(target_location=d_pos,begin_time=t,transition_time=dt)
        t0 += duration + 0.5

        duration = 3
        direction = Vector([0, 0, 8])

        s_c.move(direction=direction, begin_time=t0, transition_time=duration)
        steps = 10
        dt = duration / steps
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = geometry(a_pos, b_pos, c_new, t_pos)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bad.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            aux_line1.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            aux_line2.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            ra2.rotate(rotation_euler=[0,np.pi - alpha, 0], begin_time=t, transition_time=dt)
            ra2.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += duration + 0.5

        print("total time: ", t0)

    def addition_formal(self):
        cues = self.sub_scenes['addition_formal']
        t0 = cues['start']

        display = Display(scales=[10, 5], number_of_lines=10, location=[0, 0, 0], columns=1, flat=True)
        title = SimpleTexBObject(r"\text{Calculating with complex numbers}", color='important', aligned="center")
        display.set_title(title)

        display.appear(begin_time=t0)
        t0 += 1.5
        title.write(begin_time=t0)
        t0 += 1.5

        color1 = flatten(
            [['text'], ['drawing'], ['text'], ['example'] * 2, ['text'] * 3, ['drawing'], ['text'], ['example'] * 2,
             ['text'] * 2, ['drawing'], ['text'], ['example'] * 2, ['text'], ['drawing'], ['text'], ['example'] * 2])
        color2 = flatten([['text'], ['drawing'] * 5, ['text'], ['example'] * 6])

        line1 = SimpleTexBObject(r"(a+b\,i)+(c+d\,i)=a+b\,i+c+d\,i", color=color1)  # ,name='line1')
        line2 = SimpleTexBObject(r"=(a+c)+(b+d)\,i", color=color2)  # ,name='line2')

        indent = 2

        display.add_text_in(line1, line=1, indent=indent, column=1)
        display.add_text_in(line2, line=2, indent=indent, column=1)
        line2.align(line1, char_index=0, other_char_index=13)

        line1.write(begin_time=t0)
        t0 += 1.5
        line1.move_copy_to(target=line2, src_letter_indices=[14, 19], target_letter_indices=[2, 5], begin_time=t0,
                           transition_time=0.5)
        line2.write(letter_set=[0, 1, 3, 4], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5
        line1.move_copy_to(target=line2, src_letter_indices=[16, 21], target_letter_indices=[8, 10], begin_time=t0,
                           transition_time=0.5)
        line2.write(letter_set=[6, 7, 9, 11, 12], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5

        color3 = flatten(
            [['text'], ['drawing'], ['text'], ['example'] * 2, ['text'] * 3, ['drawing'], ['text'], ['example'] * 2,
             ['text'] * 2, ['drawing'], ['text'], ['example'] * 2, ['drawing'] * 2, ['example'] * 3])
        color4 = flatten([['text'], ['drawing'] * 5, ['text'], ['example'] * 6])

        line3 = SimpleTexBObject(r"(a+b\,i)-(c+d\,i)=a+b\,i-c-d\,i", color=color3)  # , name='line3')
        line4 = SimpleTexBObject(r"=(a-c)+(b-d)\,i", color=color4)  # , name='line4')

        display.add_text_in(line3, line=4, indent=indent, column=1)
        display.add_text_in(line4, line=5, indent=indent, column=1)
        line4.align(line3, char_index=0, other_char_index=13)

        line3.write(begin_time=t0)
        t0 += 1.5
        line3.move_copy_to(target=line4, src_letter_indices=[14, 18, 19], target_letter_indices=[2, 3, 5],
                           begin_time=t0, transition_time=0.5)
        line4.write(letter_set=[0, 1, 4], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5
        line3.move_copy_to(target=line4, src_letter_indices=[16, 20, 21], target_letter_indices=[8, 9, 10],
                           begin_time=t0, transition_time=0.5)
        line4.write(letter_set=[6, 7, 11, 12], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5

    def multiplication_formal(self):
        cues = self.sub_scenes['multiplication_formal']
        t0 = 0  # cues['start']

        display = Display(scales=[10, 5], number_of_lines=10, location=[0, 0, 0], columns=1, flat=True)
        title = SimpleTexBObject(r"\text{Calculating with complex numbers}", color='important', aligned="center")
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=0)
        title.write(begin_time=t0, transition_time=0)
        t0 += 0.5

        color1 = flatten(
            [['text'], ['drawing'], ['text'], ['example'] * 2, ['text'] * 3, ['drawing'], ['text'], ['example'] * 2,
             ['text'] * 2, ['drawing'] * 2, ['text'], ['example'] * 5, ['text'], ['drawing'], ['example'] * 2, ['text'],
             ['example'], ['drawing'], ['example']])
        color2 = flatten([['text'], ['drawing'] * 7, ['text'], ['example']])

        line1 = SimpleTexBObject(r"(a+b\,i)\cdot (c+d\,i)=ac+\,\,i^2\,\,\cdot bd+ad\,i+bc\,i",
                                 color=color1)  # ,name='line1')
        line2 = SimpleTexBObject(r"=(ac-bd)+(ad+bc)\,i", color=color2)  # ,name='line2')

        indent = 2.1

        display.add_text_in(line1, line=1, indent=indent, column=1)
        display.add_text_in(line2, line=2, indent=indent, column=1)
        line2.align(line1, char_index=0, other_char_index=13)

        line1.write(letter_range=[0, 13], begin_time=t0)
        t0 += 1.5

        line1.write(letter_range=[13, 14], begin_time=t0, transition_time=0.1)
        t0 += 0.5

        line1.move_copy_to(src_letter_indices=[1, 8], target_letter_indices=[14, 15], begin_time=t0,
                           transition_time=0.5)
        display.hide(line1, letter_set={14, 15}, begin_time=t0,
                     transition_time=0.5)  # need to hide the target letters, otherwise there will be artefacts

        line1.write(letter_set={16}, begin_time=t0 + 0.6, transition_time=0.1)
        t0 += 0.75

        line1.move_copy_to(src_letter_indices=[3, 4, 10, 11], target_letter_indices=[20, 17, 21, 17], begin_time=t0,
                           transition_time=0.5)
        display.hide(line1, letter_set={17, 20, 21}, begin_time=t0, transition_time=0.5)

        line1.write(letter_set={18}, begin_time=t0 + 0.4, transition_time=0.1)
        line1.write(letter_set={19}, begin_time=t0 + 0.5, transition_time=0.1)
        line1.write(letter_set={22}, begin_time=t0 + 0.61, transition_time=0.1)
        t0 += 0.75

        line1.move_copy_to(src_letter_indices=[1, 10, 11], target_letter_indices=[23, 24, 25], begin_time=t0,
                           transition_time=0.5)
        display.hide(line1, letter_set={23, 24, 25}, begin_time=t0, transition_time=0.5)
        line1.write(letter_set={26}, begin_time=t0 + 0.6, transition_time=0.1)
        t0 += 0.75

        line1.move_copy_to(src_letter_indices=[3, 4, 8], target_letter_indices=[27, 29, 28], begin_time=t0,
                           transition_time=0.5)
        display.hide(line1, letter_set={27, 28, 29}, begin_time=t0, transition_time=0.5)
        t0 += 0.75
        # make all letters appear
        line1.appear(letter_set={14, 15, 17, 20, 21, 23, 24, 25, 27, 28, 29}, begin_time=t0, transition_time=0)
        display.un_hide(line1, letter_set={14, 15, 17, 20, 21, 23, 24, 25, 27, 28, 29}, begin_time=t0,
                        transition_time=0.1)

        subs_colors = flatten([['drawing'] * 21, ['text'], ['example'] * 7])
        substitution = SimpleTexBObject(r"(a+b\,i)\cdot(c+d\,i)=ac+(-1)bd+ad\,i+bc\,i", color=subs_colors)

        # remove copies
        t0 += 0.1
        line1.disappear_copies(begin_time=t0, transition_time=0, display=display)

        t0 += 0.1
        #
        line1.replace(substitution, src_letter_range=[17, 19], img_letter_range=[17, 21], shift=[-0.1, 0],
                      begin_time=t0)
        t0 += 1
        line1.letters[20].change_color('drawing', begin_time=t0 + 0.1, transition_time=0.5)
        line1.letters[21].change_color('drawing', begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1
        line1.letters[23].change_color('example', begin_time=t0 + 0.1, transition_time=0.5)
        line1.letters[28].change_color('example', begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1
        line2.write(begin_time=t0)
        t0 += 1.5

        color3 = flatten(
            [['drawing'] * 2, ['text'] * 3, ['example'] * 4, ['text'] * 3, ['drawing'] * 2, ['text'] * 2,
             ['example'] * 4,
             ['text'] * 7, ['drawing'] * 2, ['text'] * 2, ['example'] * 4, ['text'] * 2])
        color4 = flatten([['text'], ['drawing'] * 11, ['text'], ['example']])

        line3 = SimpleTexBObject(r"{a+b\,i\over c+d\,i}={(a+b\,i)\cdot(c-d\,i)\over (c+d\,i)\cdot (c-d\,i)}",
                                 color=color3)
        line4 = SimpleTexBObject(r"={ac+bd\over c^2+d^2}+{bc-ad \over c^2+d^2}\,i", color=color4)

        display.add_text_in(line3, line=5, indent=indent, column=1)
        display.add_text_in(line4, line=7, indent=indent, column=1)
        line3.align(line1, char_index=9, other_char_index=19)
        line4.align(line1, char_index=0, other_char_index=19)

        line3.write(letter_range=[0, 10], begin_time=t0)
        t0 += 1.5
        line3.write(letter_range=[10, len(line3.letters)], begin_time=t0)
        t0 += 1.5
        line4.write(begin_time=t0)

        t0 += 1.5
        display.rotate(begin_time=t0, rotation_euler=[np.pi / 2, 0, np.pi])
        t0 += 1.5
        exercise = SimpleTexBObject(r"\text{Exercise}", color="important", aligned='center')
        display.set_title_back(exercise)
        exercise.write(begin_time=t0)
        t0 += 1.5

        color5 = flatten(
            [['text'], ['drawing'], ['text'], ['example'], ['text'] * 3, ['drawing'] * 2, ['text'], ['example'],
             ['text'] * 2, ['drawing']])
        color6 = flatten(
            [['text'], ['drawing'], ['text'], ['example'], ['text'] * 3, ['drawing'] * 2, ['text'], ['example'],
             ['text'] * 2, ['drawing'], ['text'], ['example']])
        color7 = flatten(
            [['text'], ['drawing'], ['text'], ['example'], ['text'] * 3, ['drawing'] * 2, ['text'], ['example'],
             ['text'] * 2, ['drawing'] * 2, ['text'], ['example']])
        color8 = flatten(
            [['drawing'] * 3, ['text'] * 3, ['example'] * 2, ['text'], ['drawing'] * 2, ['text'], ['example']])

        line5 = SimpleTexBObject(r"(3-i)+(-1+i)=2", color=color5)  # ,name='line1')
        line6 = SimpleTexBObject(r"(3-i)-(-1+i)=4-2\,i", color=color6)  # ,name='line2')
        line7 = SimpleTexBObject(r"(3-i)\,\,\cdot\,(-1+i)=-2+4\,i", color=color7)  # ,name='line2')
        line8 = SimpleTexBObject(r"{\phantom{+}3-i\over -1+i}=-2-i", color=color8)  # ,name='line2')

        indent = 2.1

        display.add_text_in_back(line5, line=1, indent=indent, column=1)
        display.add_text_in_back(line6, line=2, indent=indent, column=1)
        display.add_text_in_back(line7, line=3, indent=indent, column=1)
        display.add_text_in_back(line8, line=5, indent=indent, column=1)
        line6.align(line5, char_index=12, other_char_index=12)
        line7.align(line5, char_index=12, other_char_index=12)
        line8.align(line5, char_index=8, other_char_index=12)

        line5.write(letter_range=[0, 13], begin_time=t0)
        line6.write(letter_range=[0, 13], begin_time=t0 + 0.2)
        line7.write(letter_range=[0, 13], begin_time=t0 + 0.4)
        line8.write(letter_range=[0, 9], begin_time=t0 + 0.6)
        t0 += 1.5

        line5.write(letter_range=[13, len(line5.letters)], begin_time=t0, transition_time=0.2)
        line6.write(letter_range=[13, len(line6.letters)], begin_time=t0 + 0.2, transition_time=0.5)
        line7.write(letter_range=[13, len(line7.letters)], begin_time=t0 + 0.4, transition_time=0.5)
        line8.write(letter_range=[9, len(line8.letters)], begin_time=t0 + 0.6, transition_time=0.5)

    def complex_plane(self):
        cues = self.sub_scenes['complex_plane']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{The complex plane}", color='important', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)
        t0 += 1.5

        real_duration = 10
        tx = t0
        ty = t0 + real_duration + 3.5
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-5, 5], [-5, 5]], radii=[0.03, 0.03],
                                  all_n_tics=[5, 5],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-5, 5.1, 2), np.arange(-5, 5.1, 2)],
                                  colors=['drawing', 'example'], label_colors=['drawing', 'example'],
                                  text_size='medium', name='ComplexPlane')
        coords.appear_individually(begin_times=[tx, ty], transition_times=[3, 3])
        t0 += 3.5

        # real sphere
        digits = 1  # 1 0 for debugging
        real = SimpleTexBObject("x=\phantom{+1.0}", color='drawing', name='RealString')
        number = DigitalNumber(1, number_of_digits=digits, color=['drawing'], aligned='left', name='RealDigitalNumber')

        display.add_text_in(real, line=2, indent=1, scale=1)
        display.add_text_in(number, line=2, indent=2.5, scale=1)

        sphere = Sphere(0.25, color='drawing', location=coords.coords2location([1, 0]), smooth=2, name="RealSphere")
        sphere.grow(begin_time=t0, transition_time=0.5)

        real.write(begin_time=t0, transition_time=0.1)
        t0 += 0.1
        number.write(begin_time=t0, transition_time=0.3)
        t0 += 0.5

        t_back = t0

        sphere.move(direction=[4, 0, 0], begin_time=t0, transition_time=(real_duration - 2) / 4)  # 4
        t0 += (real_duration - 2) / 4 + 0.5
        sphere.move(direction=[-9, 0, 0], begin_time=t0, transition_time=(real_duration - 2) / 2)  # -9
        t0 += (real_duration - 2) / 2 + 0.5
        sphere.move(direction=[6, 0, 0], begin_time=t0, transition_time=(real_duration - 2) / 4)  # 6
        t0 += (real_duration - 2) / 4 + 0.5

        number.update_value(lambda frm: coords.location2coords(sphere.get_location_at_frame(frm))[0], begin_time=t_back,
                            transition_time=real_duration - 1)

        # imaginary sphere
        imag = SimpleTexBObject("y=\phantom{+1.0}", color='example', name="ImagString")
        number2 = DigitalNumber(1, number_of_digits=digits, color=['example'], aligned='left',
                                name="ImagValueDigitalNumber")

        display.add_text_in(imag, line=4, indent=1, scale=1)
        display.add_text_in(number2, line=4, indent=2.5, scale=1)

        i_sphere = Sphere(0.25, color='example', location=coords.coords2location([0, 1]), smooth=2, name='ImagSphere')
        i_sphere.grow(begin_time=t0, transition_time=0.5)

        imag.write(begin_time=t0, transition_time=0.1)
        t0 += 0.1
        number2.write(begin_time=t0, transition_time=0.3)
        t0 += 0.5

        t_back = t0

        i_sphere.move(direction=[0, 0, 1], begin_time=t0, transition_time=(real_duration - 2) / 4)
        t0 += (real_duration - 2) / 4 + 0.5
        i_sphere.move(direction=[0, 0, -3], begin_time=t0, transition_time=(real_duration - 2) / 4)
        t0 += (real_duration - 2) / 4 + 0.5
        i_sphere.move(direction=[0, 0, 6], begin_time=t0, transition_time=(real_duration - 2) / 2)
        t0 += (real_duration - 2) / 2 + 0.5

        number2.update_value(lambda frm: coords.location2coords(i_sphere.get_location_at_frame(frm))[1],
                             begin_time=t_back, transition_time=real_duration - 1)

        # complex
        complex_duration = 10
        complex = SimpleTexBObject("z=x+y\,i = +2.0+5.0\,i",
                                   color=['important', 'important', 'drawing', 'important', 'example', 'example',
                                          'important', 'example'], name='ComplexString')

        display.add_text_in(complex, line=6, indent=1, scale=1.0)

        z_real = DigitalNumber(2, number_of_digits=digits, color=['drawing'], aligned='left',
                               name="RealPartDigitalNumber")
        display.add_text_in(z_real, line=6, indent=5, scale=1)
        z_imag = DigitalNumber(5, number_of_digits=digits, color=['example'], aligned='left',
                               name="ImagPartDigitalNumber")
        display.add_text_in(z_imag, line=6, indent=6.5, scale=1)

        z_real.move_to_match_letter(target=complex, src_letter_index=2, target_letter_index=9)  # sync the decimal point
        z_imag.move_to_match_letter(target=complex, src_letter_index=2,
                                    target_letter_index=13)  # sync the decimal point

        # start at (2,5)

        mapping = lambda phi: [3 * np.sin(2 * phi) + 2, 0, 4 * np.cos(phi) + 1]
        curve = Curve(mapping, thickness=0, extrude=0, domain=[0, 2 * np.pi], name='ComplexCurve')
        curve.appear(begin_time=t0, transition_time=0)

        z_sphere = Sphere(0.25, color='important', location=[0, 0, 0], smooth=2,
                          name='FollowSphere')  # no explicit location needed for following spheres
        z_sphere.grow(begin_time=t0, transition_time=0.5)

        # z=
        complex.write(letter_range=[0, 2], begin_time=t0, transition_time=0.1)
        t0 += 0.1
        real.move_letters_to(target=complex, src_letter_indices={0, 1}, target_letter_indices={2, 6}, begin_time=t0,
                             transition_time=0.5)
        imag.move_letters_to(target=complex, src_letter_indices={0, 1}, target_letter_indices={4, 6}, begin_time=t0,
                             transition_time=0.75)

        # draw grid lines
        coords.draw_grid_lines(colors=['drawing', 'example'], begin_time=t0, transition_time=2)

        # +
        complex.write(letter_set={3}, begin_time=t0 + 0.25, transition_time=0.1)
        # i=
        complex.write(letter_set={5, 6}, begin_time=t0 + 0.75, transition_time=0.2)
        real.letters[1].hide(begin_time=t0 + 1)
        imag.letters[1].hide(begin_time=t0 + 1)
        number.move_letters_to(target=complex, src_letter_indices=(0, 1, 2, 3), target_letter_indices=(7, 8, 9, 10),
                               begin_time=t0, transition_time=0.8)
        number2.move_letters_to(target=complex, src_letter_indices=(0, 1, 2, 3), target_letter_indices=(11, 12, 13, 14),
                                begin_time=t0, transition_time=1)
        # i
        complex.write(letter_set={15}, begin_time=t0 + 1, transition_time=0.1)

        t0 += 1.5
        z_real.write(begin_time=t0, transition_time=0)
        z_imag.write(begin_time=t0, transition_time=0)

        number.hide(begin_time=t0 + 0.11)
        number2.hide(begin_time=t0 + 0.11)
        t0 += 0.5
        z_sphere.follow(curve, initial_value=0, final_value=1, begin_time=t0, transition_time=complex_duration)

        # for some reason the following lines don't work
        # z_real.update_value(lambda frm: coords.location2coords(z_sphere.get_location_at_frame(frm))[0], begin_time=t0, transition_time=complex_duration)
        # z_imag.update_value(lambda frm: coords.location2coords(z_sphere.get_location_at_frame(frm))[1], begin_time=t0, transition_time=complex_duration)
        # quick fix with separate function that obtains the position of the sphere for each frame

        z_real.update_value(
            lambda frm: coords.location2coords(get_sphere_location(z_sphere, curve, frm, mapping))[0],
            begin_time=t0, transition_time=complex_duration)
        z_imag.update_value(
            lambda frm: coords.location2coords(get_sphere_location(z_sphere, curve, frm, mapping))[1],
            begin_time=t0, transition_time=complex_duration)
        #
        sphere.update_position(lambda frm: coords.coords2location(
            [coords.location2coords(get_sphere_location(z_sphere, curve, frm, mapping))[0], 0]), begin_time=t0,
                               transition_time=complex_duration, resolution=10)
        i_sphere.update_position(lambda frm: coords.coords2location(
            [0, coords.location2coords(get_sphere_location(z_sphere, curve, frm, mapping))[1]]), begin_time=t0,
                                 transition_time=complex_duration, resolution=10)

    def addition(self):
        cues = self.sub_scenes['addition']
        t0 = 0  # cues['start']

        z1 = 3 - 1j
        z2 = -1 + 1j

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{Addition}", color='text', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 4], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 4.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[-2, 0, 0],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)
        # draw grid lines
        coords.draw_grid_lines(colors=['drawing', 'drawing'], begin_time=t0, transition_time=2, sub_grid=5)
        t0 += 3

        colors = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3, ['joker'] * 2, ['text'], ['important'], ['text'] * 3,
             ['joker'], ['text'], ['important'] * 2, ['text']])

        lines = [
            SimpleTexBObject('(' + z2str(z1) + ')+(' + z2str(z2) + ')=' + z2str(z1 + z2), aligned='left', color=colors)
        ]

        for i, line in enumerate(lines):
            display.add_text_in(line, line=i + 1, indent=1, scale=0.9)

        details = 3
        removeables = []
        sphere = Sphere(0.25, location=coords.coords2location(z2p(z1)), smooth=2, name='Sphere_A')
        removeables.append(sphere)
        coords.add_object(sphere)
        sphere.grow(begin_time=t0 + 0.5, transition_time=0.5)
        x_line_1 = Cylinder.from_start_to_end(start=coords.coords2location([0, 0]),
                                              end=coords.coords2location([re(z1), 0]), color='joker', thickness=0.5,
                                              loop_cuts=details)
        coords.add_object(x_line_1)
        removeables.append(x_line_1)
        x_line_1.grow(begin_time=t0, transition_time=0.25, modus='from_start')
        y_line_1 = Cylinder.from_start_to_end(start=coords.coords2location([re(z1), 0]),
                                              end=coords.coords2location(z2p(z1)), color='important', thickness=0.5,
                                              loop_cuts=details)
        coords.add_object(y_line_1)
        removeables.append(y_line_1)
        y_line_1.grow(begin_time=t0 + 0.25, transition_time=0.25, modus='from_start')
        t0 += 1.5
        sphere = Sphere(0.25, location=coords.coords2location(z2p(z2)), smooth=2, name='Sphere_B')
        removeables.append(sphere)
        coords.add_object(sphere)
        sphere.grow(begin_time=t0 + 0.5, transition_time=0.5)
        lines[0].write(letter_range=[6, 12], begin_time=t0, transition_time=0.5)
        x_line_2 = Cylinder.from_start_to_end(start=coords.coords2location([0, 0]),
                                              end=coords.coords2location([re(z2), 0]), color='joker',
                                              thickness=0.5, loop_cuts=details)
        coords.add_object(x_line_2)
        removeables.append(x_line_2)
        x_line_2.grow(begin_time=t0, transition_time=0.25, modus='from_start')
        y_line_2 = Cylinder.from_start_to_end(start=coords.coords2location([re(z2), 0]),
                                              end=coords.coords2location(z2p(z2)), color='important',
                                              thickness=0.5, loop_cuts=details)
        coords.add_object(y_line_2)
        removeables.append(y_line_2)
        y_line_2.grow(begin_time=t0 + 0.25, transition_time=0.25, modus='from_start')
        t0 += 1.5
        lines[0].write(letter_set={5, 12}, begin_time=t0, transition_time=0.2)
        coords.disappear_grid(begin_time=t0, transition_time=0.25)
        t0 += 0.5

        # path
        path_time = 1
        paws_add_1 = Paws(location=coords.coords2location(z2p(0)), steps=20, color='joker', scale=0.5,
                    rotation_euler=[np.pi / 2, 0, 0], name='Paw_add_1')
        paws_add_1.appear(begin_time=t0, transition_time=path_time)
        t0 += path_time + 0.1
        paws_add_2 = Paws(location=coords.coords2location(z2p(np.real(z1))), steps=6, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, np.pi / 2, 0], name='Paw_add_2')
        paws_add_2.appear(begin_time=t0, transition_time=0.3333 * path_time)
        t0 += 0.333 * path_time + 0.1
        paws_add_3 = Paws(location=coords.coords2location(z2p(z1)), steps=6, color='joker', scale=0.5,
                     rotation_euler=[np.pi / 2, np.pi, 0], name='Paw_add_3')
        paws_add_3.appear(begin_time=t0, transition_time=0.3333 * path_time)
        t0 += 0.333 * path_time + 0.1
        paws_add_4 = Paws(location=coords.coords2location(z2p(z1 + np.real(z2))), steps=6, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, -np.pi / 2, 0], name='Paw_add_4')
        paws_add_4.appear(begin_time=t0, transition_time=0.3333 * path_time)
        lines[0].write(letter_range=[13, len(lines[0].letters)], begin_time=t0, transition_time=0.5)
        t0 += 0.333 * path_time + 0.1
        coords.add_objects(paws_add_1, paws_add_2, paws_add_3, paws_add_4)
        # result
        sphere = Sphere(0.25, location=coords.coords2location(z2p(z1 + z2)), smooth=2, name='Sphere_C')
        coords.add_object(sphere)
        removeables.append(sphere)
        sphere.grow(begin_time=t0, transition_time=0.5)
        paws_add_1.disappear(begin_time=t0, transition_time=0.5)
        paws_add_2.disappear(begin_time=t0 + 0.25, transition_time=0.5)
        paws_add_3.disappear(begin_time=t0 + 0.5, transition_time=0.5)
        paws_add_4.disappear(begin_time=t0 + 0.75, transition_time=0.5)
        t0 += 2
        paws_add_1.hide(begin_time=t0 + 0.5)  # to avoid conflicts with the second path
        paws_add_2.hide(begin_time=t0 + 0.75)
        paws_add_3.hide(begin_time=t0 + 1)
        paws_add_4.hide(begin_time=t0 + 1.25)

        # arrows
        arrow_1 = Arrow.from_start_to_end(end=coords.coords2location(z2p(z1)), color='example', loop_cuts=details,
                                          name='Pfeil_1')
        arrow_2 = Arrow.from_start_to_end(end=coords.coords2location(z2p(z2)), color='example', loop_cuts=details,
                                          name='Pfeil_2')
        coords.add_objects([arrow_1, arrow_2])
        arrow_1.grow(begin_time=t0, modus='from_start')
        arrow_2.grow(begin_time=t0 + 1.5, modus='from_start')
        removeables.append(arrow_1)
        removeables.append(arrow_2)
        t0 += 3
        cp_arrow = arrow_2.move_copy_to(target_location=coords.coords2location(z2p(z1)), begin_time=t0)
        cp_x_line_2 = x_line_2.move_copy(direction=coords.coords2location(z2p(z1)), begin_time=t0)
        cp_y_line_2 = y_line_2.move_copy(direction=coords.coords2location(z2p(z1)), begin_time=t0)

        removeables.append(cp_arrow)
        removeables.append(cp_x_line_2)
        removeables.append(cp_y_line_2)

        t0 += 1.5

        explosion = Explosion(removeables)
        explosion.set_wind_and_turbulence(wind_location=[6, 0, -5], turbulence_location=[0, 0, 0],
                                          rotation_euler=[0, -np.pi / 4, 0], wind_strength=1.5, turbulence_strength=10)
        explosion.explode(begin_time=t0, transition_time=2)

        ############### Subtraction ################

        title = SimpleTexBObject(r"\text{Subtraction}", color='text', aligned="center", name='Title')
        display.set_title(title, shift=[-2.5,0])
        t0 += 1
        title.write(begin_time=t0, transition_time=1)
        t0 += 1.5

        colors = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3, ['joker'] * 2, ['text'], ['important'], ['text'] * 3,
             ['joker'], ['text'], ['important'] * 2, ['text']])
        lines.append(
            SimpleTexBObject('(' + z2str(z1) + ')-(' + z2str(z2) + ')=(' + z2str(z1 - z2) + ')', aligned='left',
                             color=colors)
        )

        for i in range(1, 2):
            display.add_text_in(lines[i], line=i + 4, indent=1, scale=0.9)

        lines[-1].write(letter_range=[0, 5], begin_time=t0, transition_time=1)

        details = 3
        removeables = []
        sphere = Sphere(0.25, location=coords.coords2location(z2p(z1)), smooth=2, name='Sphere_A')
        removeables.append(sphere)
        coords.add_object(sphere)
        sphere.grow(begin_time=t0 + 0.5, transition_time=0.5)
        x_line_1 = Cylinder.from_start_to_end(start=coords.coords2location([0, 0]),
                                              end=coords.coords2location([re(z1), 0]), color='joker', thickness=0.5,
                                              loop_cuts=details)
        coords.add_object(x_line_1)
        removeables.append(x_line_1)
        x_line_1.grow(begin_time=t0, transition_time=0.25, modus='from_start')
        y_line_1 = Cylinder.from_start_to_end(start=coords.coords2location([re(z1), 0]),
                                              end=coords.coords2location(z2p(z1)), color='important', thickness=0.5,
                                              loop_cuts=details)
        coords.add_object(y_line_1)
        removeables.append(y_line_1)
        y_line_1.grow(begin_time=t0 + 0.25, transition_time=0.25, modus='from_start')
        t0 += 1.5
        sphere = Sphere(0.25, location=coords.coords2location(z2p(z2)), smooth=2, name='Sphere_B')
        removeables.append(sphere)
        coords.add_object(sphere)
        sphere.grow(begin_time=t0 + 0.5, transition_time=0.5)
        lines[-1].write(letter_range=[6, 12], begin_time=t0, transition_time=0.5)
        x_line_2 = Cylinder.from_start_to_end(start=coords.coords2location([0, 0]),
                                              end=coords.coords2location([re(z2), 0]), color='joker',
                                              thickness=0.5, loop_cuts=details)
        coords.add_object(x_line_2)
        removeables.append(x_line_2)
        x_line_2.grow(begin_time=t0, transition_time=0.25, modus='from_start')
        y_line_2 = Cylinder.from_start_to_end(start=coords.coords2location([re(z2), 0]),
                                              end=coords.coords2location(z2p(z2)), color='important',
                                              thickness=0.5, loop_cuts=details)
        coords.add_object(y_line_2)
        removeables.append(y_line_2)
        y_line_2.grow(begin_time=t0 + 0.25, transition_time=0.25, modus='from_start')
        t0 += 1.5
        lines[-1].write(letter_set={5, 12}, begin_time=t0, transition_time=0.2)
        t0 += 0.5

        # path
        path_time = 1
        paws = Paws(location=coords.coords2location(z2p(0)), steps=20, color='joker', scale=0.5,
                    rotation_euler=[np.pi / 2, 0, 0], name='Paw_sub_1')
        paws.appear(begin_time=t0, transition_time=path_time)
        paws.hide(begin_time=0)  # to avoid conflict with first path
        paws.un_hide(begin_time=t0)
        t0 += path_time + 0.1
        paws2 = Paws(location=coords.coords2location(z2p(np.real(z1))), steps=6, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, np.pi / 2, 0], name='Paw_sub_2')
        paws2.appear(begin_time=t0, transition_time=0.3333 * path_time)
        paws2.hide(begin_time=0)
        paws2.un_hide(begin_time=t0)
        t0 += 0.333 * path_time + 0.1
        paws3 = Paws(location=coords.coords2location(z2p(z1)), steps=6, color='joker', scale=0.5,
                     rotation_euler=[np.pi / 2, 0, 0], name='Paw_sub_3')
        paws3.appear(begin_time=t0, transition_time=0.3333 * path_time)
        paws3.hide(begin_time=0)
        paws3.un_hide(begin_time=t0)
        t0 += 0.333 * path_time + 0.1
        paws4 = Paws(location=coords.coords2location(z2p(z1 - np.real(z2))), steps=6, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, np.pi / 2, 0], name='Paw_sub_4')
        paws4.appear(begin_time=t0, transition_time=0.3333 * path_time)
        paws4.hide(begin_time=0)
        paws4.un_hide(begin_time=t0)
        lines[1].write(letter_range=[13, len(lines[1].letters)], begin_time=t0, transition_time=1)
        t0 += 0.333 * path_time + 0.1
        coords.add_objects(paws, paws2, paws3, paws4)
        # result
        sphere = Sphere(0.25, location=coords.coords2location(z2p(z1 - z2)), smooth=2, name='Sphere_C')
        coords.add_object(sphere)
        removeables.append(sphere)
        sphere.grow(begin_time=t0, transition_time=0.5)
        paws.disappear(begin_time=t0, transition_time=0.5)
        paws2.disappear(begin_time=t0 + 0.25, transition_time=0.5)
        paws3.disappear(begin_time=t0 + 0.5, transition_time=0.5)
        paws4.disappear(begin_time=t0 + 0.75, transition_time=0.5)
        paws4.hide(begin_time=t0 + 1.25)
        paws3.hide(begin_time=t0 + 1.0)
        paws2.hide(begin_time=t0 + 0.75)
        paws.hide(begin_time=t0 + 0.5)

        t0 += 2

        # arrows
        arrow_3 = Arrow.from_start_to_end(end=coords.coords2location(z2p(z1)), color='example', loop_cuts=details,
                                          name='Pfeil_3')
        arrow_4 = Arrow.from_start_to_end(end=coords.coords2location(z2p(z2)), color='example', loop_cuts=details,
                                          name='Pfeil_4')
        coords.add_objects([arrow_3, arrow_4])
        arrow_3.grow(begin_time=t0, modus='from_start')
        arrow_4.grow(begin_time=t0 + 0.4, modus='from_start')
        removeables.append(arrow_3)
        removeables.append(arrow_4)
        t0 += 3

        cp_arrow = arrow_4.copy()
        cp_x_line_2 = x_line_2.copy()
        cp_arrow.add_child(cp_x_line_2)
        cp_y_line_2 = y_line_2.copy()
        cp_arrow.add_child(cp_y_line_2)
        cp_arrow.rotate(rotation_euler=[0, np.pi, 0], begin_time=t0)
        cp_arrow.move_to(target_location=coords.coords2location(z2p(z1)), begin_time=t0 + 1)

        removeables.append(cp_arrow)
        removeables.append(cp_x_line_2)
        removeables.append(cp_y_line_2)

        t0 += 2.5

        explosion = Explosion(removeables)
        explosion.set_wind_and_turbulence(wind_location=[6, 0, -5], turbulence_location=[0, 0, 0],
                                          rotation_euler=[0, -np.pi / 4, 0], wind_strength=1.5, turbulence_strength=10)
        explosion.explode(begin_time=t0, transition_time=2)
        t0+=3
        print("Ende ",t0*FRAME_RATE)

    def scaling(self):
        cues = self.sub_scenes['scaling']
        t0=0

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{Addition}", color='text', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=0)
        title.write(begin_time=t0, transition_time=0)

        title2 = SimpleTexBObject(r"\text{Subtraction}", color='text', aligned="center", name='Title')
        display.set_title(title2, shift=[-2.5, 0])
        title2.write(begin_time=t0, transition_time=0)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 4], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 4.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[-2, 0, 0],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)

        z1 = 3 - 1j
        z2 = -1 + 1j

        colors = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3, ['joker'] * 2, ['text'], ['important'], ['text'] * 3,
             ['joker'], ['text'], ['important'] * 2, ['text']])
        colors2 = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3, ['joker'] * 2, ['text'], ['important'], ['text'] * 3,
             ['joker'], ['text'], ['important'] * 2, ['text']])
        lines = [
            SimpleTexBObject('(' + z2str(z1) + ')+(' + z2str(z2) + ')=' + z2str(z1 + z2), aligned='left', color=colors),
            SimpleTexBObject('(' + z2str(z1) + ')-(' + z2str(z2) + ')=(' + z2str(z1 - z2) + ')', aligned='left',
                             color=colors2)
        ]

        rows=[1,5]
        for index, line in enumerate(lines):
            display.add_text_in(line, line=rows[index], indent=1, scale=0.9)
            line.write(begin_time=t0,transition_time=0)

        ############## Scaling ##################
        title3 = SimpleTexBObject(r"\text{Scaling}", color='text', aligned="center", name='Title')
        display.set_title(title3, shift=[-5,0])
        t0 += 1
        title3.write(begin_time=t0, transition_time=1)
        t0 += 3

        z3 = 2 + 1j
        scalar = SimpleTexBObject(r"{1 \over 2}\,\,\,", color='text')# start with most complex string to avoid null curve generation
        lines.append(scalar)

        colors = ['text', 'text', 'joker', 'text', 'important', 'text']
        lines.append(
            SimpleTexBObject(r'\,\,\cdot\,\,(' + z2str(z3) + ')', color=colors)
        )
        display.set_cursor_to_start_of_line(9, indent=1)
        for index in range(2, 4):
            display.add_text(lines[index], scale=0.9)

        scalar.write(begin_time=t0-0.1, transition_time=0)
        scalar.hide(begin_time=t0-0.1)

        scalar.add_to_morph_chain(SimpleTexBObject("1\,\,\,", color='text'),
                                  [0, 3], [0, 1], [1] * 3, [0] * 3, ['text'], ['text'], begin_time=t0,
                                  transition_time=0)
        scalar.appear(begin_time=t0,transition_time=0.5)
        z3_sphere = Sphere(0.25, location=coords.coords2location(z2p(z3)), color='drawing', name='Z3Sphere')
        z3_sphere.grow(begin_time=t0)
        lines[-1].write(begin_time=t0 + 0.5, transition_time=1)

        # path times 1
        path_time = 1
        paws = Paws(location=coords.coords2location(z2p(0)), steps=12, color='joker', scale=0.5,
                    rotation_euler=[np.pi / 2, 0, 0], name='Paw_scal_1')
        paws.appear(begin_time=t0, transition_time=path_time)
        paws.hide(begin_time=0)
        paws.un_hide(begin_time=t0)

        t0 += path_time + 0.5
        paws2 = Paws(location=coords.coords2location(z2p(np.real(z3))), steps=6, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, - np.pi / 2, 0], name='Paw_scal_2')
        paws2.appear(begin_time=t0, transition_time=0.666 * path_time)
        t0 += 0.666 * path_time + 0.5

        paws.disappear(begin_time=t0, transition_time=1)
        paws2.disappear(begin_time=t0 + 0.5, transition_time=1)
        paws.hide(begin_time=t0 + 1)
        paws2.hide(begin_time=t0 + 1.5)
        t0 += 2

        # arrows
        arrow_f = PArrow(end=coords.coords2location(z2p(z3)), color='example', loop_cuts=0)
        arrow_f.grow(begin_time=t0)
        t0 += 1

        # path times 2
        scalar.add_to_morph_chain(SimpleTexBObject("2\,\,\,", color='text'),
                                  [0, 3], [0, 1], [1] * 3, [0] * 3, ['text'], ['text'], begin_time=t0,
                                  transition_time=0.5)
        path_time = 1
        paws3 = Paws(location=coords.coords2location(z2p(0)), steps=26, color='joker', scale=0.5,
                     rotation_euler=[np.pi / 2, 0, 0], name='Paw_scal_3')
        paws3.appear(begin_time=t0, transition_time=path_time)
        paws3.hide(begin_time=0)
        paws3.un_hide(begin_time=t0)

        t0 += path_time + 0.5
        paws4 = Paws(location=coords.coords2location(z2p(np.real(2 * z3))), steps=14, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, - np.pi / 2, 0], name='Paw_scal_4')
        paws4.appear(begin_time=t0, transition_time=0.666 * path_time)
        paws4.hide(begin_time=0)
        paws4.un_hide(begin_time=t0)
        t0 += 0.666 * path_time + 0.5

        # 1->2
        z3_sphere.move_to(target_location=coords.coords2location(z2p(2 * z3)), begin_time=t0)
        arrow_f.rescale(rescale=[2, 2, 2], begin_time=t0)
        t0 += 1.5

        paws3.disappear(begin_time=t0, transition_time=1)
        paws4.disappear(begin_time=t0 + 0.5, transition_time=1)
        paws3.hide(begin_time=t0 + 1)
        paws4.hide(begin_time=t0 + 1.5)
        t0 += 2

        # path times 1/2
        scalar.add_to_morph_chain(SimpleTexBObject(r"{1 \over 2}\,\,\,", color='text'),
                                  [0, 3], [0, 3], [1] * 3, [0] * 3, ['text'], ['text'], begin_time=t0,
                                  transition_time=0.5)
        path_time = 1
        paws5 = Paws(location=coords.coords2location(z2p(0)), steps=6, color='joker', scale=0.5,
                     rotation_euler=[np.pi / 2, 0, 0], name='Paw_scal_5')
        paws5.appear(begin_time=t0, transition_time=path_time)
        paws5.hide(begin_time=0)
        paws5.un_hide(begin_time=t0)

        t0 += path_time + 0.5
        paws6 = Paws(location=coords.coords2location(z2p(np.real(0.5 * z3))), steps=3, color='important', scale=0.5,
                     rotation_euler=[np.pi / 2, - np.pi / 2, 0], name='Paw_scal_4')
        paws6.appear(begin_time=t0, transition_time=0.666 * path_time)
        paws6.hide(begin_time=0)
        paws6.un_hide(begin_time=t0)
        t0 += 0.666 * path_time + 0.5

        # 2->1/2
        z3_sphere.move_to(target_location=coords.coords2location(z2p(0.5 * z3)), begin_time=t0)
        arrow_f.rescale(rescale=[0.25, 0.25, 0.25], begin_time=t0)
        t0 += 1.5

        paws5.disappear(begin_time=t0, transition_time=1)
        paws6.disappear(begin_time=t0 + 0.5, transition_time=1)
        paws5.hide(begin_time=t0 + 1)
        paws6.hide(begin_time=t0 + 1.5)
        t0 += 2

        # 1/2 -> -1
        scalar.add_to_morph_chain(SimpleTexBObject(r"{-1}\,\,\,", color='text'),
                                  [0, 3], [0, 2], [1] * 3, [-0.25,0,0] , ['text'], ['text'], begin_time=t0,
                                  transition_time=0.5)

        z3_sphere.move_to(target_location=coords.coords2location(z2p(- z3)), begin_time=t0)
        arrow_f.rescale(rescale=[-2, -2, -2], begin_time=t0)

        t0 += 1.5

        # arbitrary
        colors = flatten([['drawing'], ['text'] * 2, ['joker'],
                          ['important', 'important', 'text']])

        lines.append(
            SimpleTexBObject(r'\lambda\,\,\cdot\,\,(' + z2str(z3) + ')', aligned='left', color=colors)
        )
        display.set_cursor_to_start_of_line(11, indent=1)
        display.add_text(lines[4], scale=0.9)
        lines[4].write(begin_time=t0, transition_time=0.5)

        labels = [
            r"\lambda=-1", r"\lambda=-{1\over 2}", r"\lambda=0",
            r"\lambda={1\over 2}", r"\lambda=-1", r"\lambda={3\over 2}", r"\lambda=2"
        ]
        count = 0
        duration = 3
        arrow_f.rescale(rescale=[-2, -2, -2], begin_time=t0, transition_time=duration)
        z3_sphere.move_to(target_location=coords.coords2location(z2p(2.01 * z3)), begin_time=t0,
                          transition_time=duration)

        l_old = np.Infinity

        coords.add_object(z3_sphere)
        for frame in range(int(t0 * FRAME_RATE), int((t0 + duration) * FRAME_RATE) + 1):
            location = coords.ref_obj.matrix_world @ ibpy.get_location_at_frame(z3_sphere, frame)
            z = p2z(coords.location2coords(location))
            l = np.floor(2 * np.sign(np.real(z)) * np.abs(z) / np.abs(z3)) / 2
            if l_old != l:
                l_old = l
                sphere = Sphere(0.25, location=coords.coords2location(z2p(l * z3)), name=labels[count], color='drawing')
                t = frame / FRAME_RATE
                sphere.grow(begin_time=t, transition_time=0.1)
                if count < 2:
                    pos='down'
                else:
                    pos = 'up_left'
                count += 1
                sphere.write_name_as_label(modus=pos, begin_time=t + 0.1, transition_time=0.2)
                coords.add_object(sphere)

        coords.add_objects([paws, paws2, paws3, paws4, paws5, paws6, arrow_f])
        scalar.perform_morphing()
        t0 += duration + 1

        arrow_f.disappear(begin_time=t0)
        line = Cylinder.from_start_to_end(start=coords.coords2location(z2p(-2 * z3)),
                                          end=coords.coords2location(z2p(3 * z3)), color='drawing', thickness=0.5)
        coords.add_object(line)
        line.grow(modus='from_start', begin_time=t0)

        t0 += 1.5
        print("ende:",t0*FRAME_RATE)

    def complex_plane2(self):
        cues = self.sub_scenes['complex_plane2']
        t0 = 0  # cues['start']

        angle = np.pi / 8
        ibpy.set_camera_location(location=[0, 0, 0])
        circle_a = Curve(lambda phi: [6, -20 * np.cos(phi), 20 * np.sin(phi)], domain=[0, angle],
                         thickness=0, name='circle_a')
        ibpy.set_camera_follow(target=circle_a)

        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{The complex plane}", color='text', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)
        t0 += 1.5

        real_duration = 5
        tx = t0
        ty = t0 + real_duration + 4
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-3, 3], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-3, 3.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small', name='ComplexPlane')
        coords.appear_individually(begin_times=[tx, ty], transition_times=[3, 3])
        t0 += 3.5

        # real sphere
        digits = 1  # 1 0 for debugging
        real = SimpleTexBObject("x=\phantom{+1.0}", color='joker', name='RealString')
        number = DigitalNumber(1, number_of_digits=digits, color=['joker'], aligned='left', name='RealDigitalNumber')

        display.add_text_in(real, line=2, indent=1, scale=1)
        display.add_text_in(number, line=1.95, indent=2.5, scale=1)

        sphere = Sphere(0.25, color='joker', location=coords.coords2location([1, 0]), smooth=2, name="RealSphere")
        sphere.grow(begin_time=t0, transition_time=0.5)

        real.write(begin_time=t0, transition_time=0.1)
        t0 += 0.1
        number.write(begin_time=t0, transition_time=0.3)
        t0 += 0.5

        t_back = t0

        sphere.move(direction=coords.coords2location([-3, 0]), begin_time=t0, transition_time=1.5)
        t0 += 2
        sphere.move(direction=coords.coords2location([5, 0]), begin_time=t0, transition_time=2.5)
        t0 += 3

        number.update_value(lambda frm: coords.location2coords(sphere.get_location_at_frame(frm))[0], begin_time=t_back,
                            transition_time=5)

        # imaginary sphere
        imag = SimpleTexBObject("y=\phantom{+1.0}\,\,\, i", color='important', name="ImagString")
        number2 = DigitalNumber(1, number_of_digits=digits, color=['important'], aligned='left',
                                name="ImagValueDigitalNumber")

        display.add_text_in(imag, line=4, indent=1, scale=1)
        display.add_text_in(number2, line=3.95, indent=2.5, scale=1)

        i_sphere = Sphere(0.25, color='important', location=coords.coords2location([0, 1]), smooth=2, name='ImagSphere')
        i_sphere.grow(begin_time=t0, transition_time=0.5)
        coords.add_object(i_sphere)

        imag.write(begin_time=t0, transition_time=0.1)
        t0 += 0.1
        number2.write(begin_time=t0, transition_time=0.3)
        t0 += 0.5

        t_back = t0

        i_sphere.move(direction=coords.coords2location([0, -4]), begin_time=t0, transition_time=3)
        t0 += 3.5
        i_sphere.move(direction=coords.coords2location([0, 2]), begin_time=t0, transition_time=1)
        t0 += 1.5

        number2.update_value(lambda frm: coords.location2coords(i_sphere.get_location_at_frame(frm))[1],
                             begin_time=t_back, transition_time=real_duration)

        detail = 2
        # draw grid lines
        coords.draw_grid_lines(loop_cuts=detail, colors=['drawing', 'drawing'], begin_time=t0, transition_time=2,
                               sub_grid=5)
        t0 += 2.5

        # complex
        complex = SimpleTexBObject("z = 3-i",
                                   color=['drawing'],
                                   name='ComplexString')  # , 'drawing', 'joker', 'drawing', 'important', 'important','drawing', 'joker','important','important'

        display.add_text_in(complex, line=6, indent=1, scale=1.0)

        complex.write(letter_range=[0, 5], begin_time=t0, transition_time=1)

        t0 += 0.5

        real_cyl = Cylinder.from_start_to_end(loop_cuts=detail, start=coords.coords2location([0, 0]),
                                              end=coords.coords2location([3, 0]), color='joker', thickness=0.5)
        coords.add_object(real_cyl)
        real_cyl.grow(begin_time=t0, transition_time=1, modus='from_start')
        t0 += 1.5

        im_cyl = Cylinder.from_start_to_end(loop_cuts=detail, start=coords.coords2location([3, 0]),
                                            end=coords.coords2location([3, -1]), color='important', thickness=0.5)
        coords.add_object(im_cyl)
        im_cyl.grow(begin_time=t0, transition_time=0.5, modus='from_start')
        t0 += 1

        z_sphere = Sphere(0.25, location=coords.coords2location([3, -1]), color='drawing', name="3-i")
        z_sphere.grow(begin_time=t0, transition_time=0.5)
        coords.add_object(z_sphere)
        t0 += 0.5
        z_sphere.write_name_as_label(modus='down_right', begin_time=t0, transition_time=0.5)
        t0 += 1

        print("start of explosion", t0)
        removeables = flatten([coords.x_lines, coords.y_lines, [real_cyl, im_cyl, sphere, i_sphere]])
        explosion = Explosion(removeables)
        explosion.explode(begin_time=t0, transition_time=5)
        t0 += 5

        display.disappear(begin_time=t0, transition_time=2)
        ibpy.camera_follow(circle_a, initial_value=0, final_value=1, begin_time=t0, transition_time=2)
        ibpy.camera_move(shift=[-6, 0, 0], begin_time=t0, transition_time=2)
        empty.move(direction=[-6, 0, 0], begin_time=t0, transition_time=2)
        t0 += 2

        stage = BObject.from_file('Stage', rotation_euler=[0, 0, -np.pi / 2], location=[0, -0.6, -0.31],
                                  scale=[1.5, 0.75, 0.02], color='gray_1')
        stage.appear(begin_time=t0, transition_time=1)
        stage.set_rigid_body(dynamic=False)  # make stage interact with curtain but do not feel gravity

        curtain = Curtain(scale=[4.5, 3.8, 1], location=[-4.5, -5.25, 3.9], number_of_hooks=15,
                          simulation_start=t0 - 2, simulation_duration=30, shadow=False)
        curtain.move(direction=[0, 0, -0.3], begin_time=t0 - 1)
        curtain.appear(begin_time=t0, transition_time=4)
        curtain.open(fraction=0.3, begin_time=t0 + 1, transition_time=2)

        curtain2 = Curtain(scale=[4.5, 3.8, 1], location=[4.5, -5.65, 3.9], rotation_euler=[np.pi / 2, 0, np.pi],
                           number_of_hooks=15,
                           simulation_start=t0 - 2, simulation_duration=30, shadow=False)
        curtain2.move(direction=[0, 0, -0.3], begin_time=t0 - 1)
        curtain2.appear(begin_time=t0, transition_time=4)
        curtain2.open(fraction=0.3, begin_time=t0 + 1, transition_time=2)

        curtain3 = Curtain(scale=[9, 1, 1], location=[0, -6.1, 7.5], rotation_euler=[np.pi / 2, 0, 0],
                           number_of_hooks=30, simulation_start=t0 - 2, simulation_duration=30, shadow=False)
        curtain3.appear(begin_time=t0, transition_time=3)
        curtain3.open(fraction=0.9, begin_time=t0 - 1, transition_time=0.5)

        energy = 2500
        coords.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=t0, transition_time=2)
        tilt = np.pi / 2 - angle
        pos = coords.coords2location([3, -1])
        z_sphere.label_disappear(begin_time=t0 + 1, transition_time=0.1)
        t0 += 4

        a = Person(color='drawing', location=[pos.x, pos.z, -0.75], scale=0.25, name=r'3-i\,')
        a.appear(begin_time=t0, transition_time=0)
        spot_a = SpotLight(target=a, energy=0.75 * energy, location=[0, -4.5, 7], name='SpotA', radius=1)
        spot_a.on(begin_time=t0, transition_time=0.5)
        a.write_name_as_label(begin_time=t0 + 0.5, transition_time=1, modus='up', offset=[0, 0, 4], scale=4,
                              rotation_euler=[tilt, 0, 0])
        z_sphere.disappear(begin_time=t0 + 0.5, transition_time=0)
        a.move(direction=[0, 0, 0.75], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5

        zs = [3 - 1j, -2 + 2j, -1 - 3j]
        positions = []
        for z in zs:
            positions.append(coords.coords2location([np.real(z), np.imag(z)]))

        persons = [a]

        name = z2str(zs[1])
        person = Person(name=name, location=positions[0], rotation_euler=[np.pi / 2, 0, 0], scale=0.25)
        coords.add_object(person)
        persons.append(person)
        person.grow(begin_time=t0, transition_time=0.5)
        person.move_to(target_location=positions[1], begin_time=t0 + 0.4)
        spot = SpotLight(target=person, energy=energy, location=[0, -4.5, 7], name='Spot_' + str(1), radius=1)
        spot.on(begin_time=t0 - 1, transition_time=0.5)
        person.write_name_as_label(rotation_euler=[tilt, 0, 0], modus='up', offset=[0, 0, 4], scale=4,
                                   begin_time=t0 + 1.5, transition_time=0.5)
        t0 += 2

        name = z2str(zs[2])
        person2 = Person(name=name, location=[5, -1.667, 0],
                         rotation_euler=[0, 0, 0], scale=0.25)
        persons.append(person2)
        person2.grow(begin_time=t0, transition_time=0.5)
        empty = EmptyCube(location=[0, 0, 0])
        rotating_object = BObject(children=[empty, person2], name='Rotation_Box')
        rotating_object.appear(begin_time=t0, transition_time=0)
        rotating_object.rotate(pivot=[0, 0, 0], rotation_euler=[0, 0, -np.pi / 2], begin_time=t0 + 0.4)
        spot2 = SpotLight(target=person2, energy=energy, location=[0, -4.5, 7], name='Spot_' + str(2), radius=1)
        spot2.on(begin_time=t0 - 1, transition_time=0.5)
        person2.write_name_as_label(rotation_euler=[tilt, 0, np.pi / 2], modus='up', offset=[0, 0, 4], scale=4,
                                    begin_time=t0 + 1.5, transition_time=0.5)
        t0 += 2

        t0 += 3
        curtain.close(begin_time=t0, transition_time=5)
        curtain2.close(begin_time=t0, transition_time=5)

    def complex_plane_walking(self):
        cues = self.sub_scenes['complex_plane_walking']
        t0 = 0  # cues['start']

        angle = np.pi / 8
        ibpy.set_camera_location(location=[0, 0, 0])
        circle_a = Curve(lambda phi: [6, -20 * np.cos(phi), 20 * np.sin(phi)], domain=[0, angle],
                         thickness=0, name='circle_a')
        ibpy.set_camera_follow(target=circle_a)

        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{The complex plane}", color='text', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=0)
        title.write(begin_time=t0, transition_time=0)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-3, 3], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-3, 3.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small', name='ComplexPlane')
        coords.appear(begin_time=0, transition_time=0)

        # real sphere
        digits = 1 # 1 0 for debugging
        real = SimpleTexBObject("x=\phantom{+1.0}", color='joker', name='RealString')
        number = DigitalNumber(3, number_of_digits=digits, color=['joker'], aligned='left', name='RealDigitalNumber')

        display.add_text_in(real, line=2, indent=1, scale=1)
        display.add_text_in(number, line=1.95, indent=2.5, scale=1)

        sphere = Sphere(0.25, color='joker', location=coords.coords2location([3, 0]), smooth=2, name="RealSphere")
        sphere.grow(begin_time=t0, transition_time=0)

        real.write(begin_time=t0, transition_time=0.0)
        number.write(begin_time=t0, transition_time=0.0)

        # imaginary sphere
        imag = SimpleTexBObject("y=\phantom{+1.0}\,\,\, i", color='important', name="ImagString")
        number2 = DigitalNumber(-1, number_of_digits=digits, color=['important'], aligned='left',
                                name="ImagValueDigitalNumber")

        display.add_text_in(imag, line=4, indent=1, scale=1)
        display.add_text_in(number2, line=3.95, indent=2.5, scale=1)

        i_sphere = Sphere(0.25, color='important', location=coords.coords2location([0, -1]), smooth=2,
                          name='ImagSphere')
        i_sphere.grow(begin_time=t0, transition_time=0)
        coords.add_object(i_sphere)
        imag.write(begin_time=t0, transition_time=0.0)
        number2.write(begin_time=t0, transition_time=0.0)

        coords.draw_grid_lines(loop_cuts=0, colors=['drawing', 'drawing'], begin_time=t0, transition_time=0, sub_grid=5)
        complex = SimpleTexBObject("z = 3-i",
                                   color=['drawing'],
                                   name='ComplexString')  # , 'drawing', 'joker', 'drawing', 'important', 'important','drawing', 'joker','important','important'

        display.add_text_in(complex, line=6, indent=1, scale=1.0)

        complex.write(letter_range=[0, 5], begin_time=t0, transition_time=0)
        real_cyl = Cylinder.from_start_to_end(loop_cuts=0, start=coords.coords2location([0, 0]),
                                              end=coords.coords2location([3, 0]), color='joker', thickness=0.5)
        coords.add_object(real_cyl)
        real_cyl.grow(begin_time=t0, transition_time=0, modus='from_start')

        im_cyl = Cylinder.from_start_to_end(loop_cuts=0, start=coords.coords2location([3, 0]),
                                            end=coords.coords2location([3, -1]), color='important', thickness=0.5)
        coords.add_object(im_cyl)
        im_cyl.grow(begin_time=t0, transition_time=0, modus='from_start')

        z_sphere = Sphere(0.25, location=coords.coords2location([3, -1]), color='drawing', name="3-i")
        z_sphere.grow(begin_time=t0, transition_time=0)
        coords.add_object(z_sphere)
        z_sphere.write_name_as_label(modus='down_right', begin_time=t0, transition_time=0)

        t0 += 1
        # paws = Paws(location=coords.coords2location(z2p(0)), steps=20, color='joker', scale=0.5,
        #             rotation_euler=[np.pi / 2, 0, 0], name='Paw_3')
        # paws.appear(begin_time=t0, transition_time=2)
        t0 += 3
        # paws.disappear(begin_time=t0, transition_time=0.5)
        # paws.hide(begin_time=t0 + 0.5)

        t0 += 1
        # paws_pos = Paws(location=coords.coords2location(z2p(0)), steps=10, color='joker', scale=0.5,
        #                 rotation_euler=[np.pi / 2, 0, 0], name='Paw_pos')
        # paws_pos.appear(begin_time=t0, transition_time=1)
        # paws_pos.hide(begin_time=0)
        # paws_pos.un_hide(begin_time=t0)
        t0 += 1.5
        # paws_pos.disappear(begin_time=t0, transition_time=0.5)
        # paws_pos.hide(begin_time=t0 + 0.5)

        t0 += 1
        # paws_neg = Paws(location=coords.coords2location(z2p(0)), steps=10, color='joker', scale=0.5,
        #                 rotation_euler=[np.pi / 2, np.pi, 0], name='Paw_neg')
        # paws_neg.appear(begin_time=t0, transition_time=1)
        t0 += 1.5
        # paws_neg.disappear(begin_time=t0, transition_time=0.5)

        t0 += 1
        # paws = Paws(location=coords.coords2location(z2p(3)), steps=6, color='important', scale=0.5,
        #             rotation_euler=[np.pi / 2, np.pi / 2, 0], name='Paw_m1')
        # paws.appear(begin_time=t0, transition_time=2)
        t0 += 3
        # paws.disappear(begin_time=t0, transition_time=0.5)
        # paws.hide(begin_time=t0 + 0.5)

        t0 += 1
        # paws_pos = Paws(location=coords.coords2location(z2p(3)), steps=10, color='important', scale=0.5,
        #                 rotation_euler=[np.pi / 2, -np.pi / 2, 0], name='Paw_pos')
        # paws_pos.appear(begin_time=t0, transition_time=1)
        # paws_pos.hide(begin_time=0)
        # paws_pos.un_hide(begin_time=t0)
        t0 += 1.5
        # paws_pos.disappear(begin_time=t0, transition_time=0.5)
        # paws_pos.hide(begin_time=t0 + 0.5)

        t0 += 1
        paws_neg = Paws(location=coords.coords2location(z2p(3)), steps=6, color='important', scale=0.5,
                        rotation_euler=[np.pi / 2, np.pi / 2, 0], name='Paw_neg')
        paws_neg.appear(begin_time=t0, transition_time=0.5)

        t0+=1.5
        # replace
        paws_neg.disappear(begin_time=t0, transition_time=0.5)

        duration = 14
        mapping = lambda phi: [5 * np.cos(phi), 0, -3.3333 * np.sin(2 * phi) - 1.667]

        f_sphere = Sphere(0.25, color='drawing', location=[5,0,-1.666], smooth=2,
                          name='FollowSphere')  # no explicit location needed for following spheres
        f_sphere.grow(begin_time=t0, transition_time=0.5)

        real_cyl.move(direction=coords.coords2location([0, -1]), begin_time=t0, transition_time=0.4)
        line_x2 = Cylinder.from_start_to_end(start=coords.coords2location([0, 0]), end=coords.coords2location([3, 0]),
                                             color='joker', thickness=0.5,name='LineX2')
        line_x2.grow(modus='from_center', begin_time=t0, transition_time=0)

        im_cyl.move(direction=coords.coords2location([-3,0]), begin_time=t0, transition_time=0.4)
        line_y2 = Cylinder.from_start_to_end(start=coords.coords2location([0, 0]), end=coords.coords2location([0, -1]),
                                             color='important', thickness=0.5,name='LineY2')
        line_y2.grow(modus='from_center', begin_time=t0, transition_time=0)

        line_x = Cylinder.from_start_to_end(start=coords.coords2location([0, -1]), end=coords.coords2location([3, -1]),
                                            color='joker', thickness=0.5,name='LineX')
        line_x.grow(modus='from_center', begin_time=t0 + 0.4, transition_time=0)

        # re_ra = RightAngle(location=[0, 0, -1.666], rotation_euler=[np.pi / 2, 0, 0], color='joker', thickness=0.25,
        #                    name='ra_re', radius=0.5)
        # re_ra.appear(begin_time=t0+0.5, transition_time=0.5)

        line_y = Cylinder.from_start_to_end(start=coords.coords2location([3, 0]), end=coords.coords2location([3, -1]),
                                            color='important', thickness=0.5,name='LineY')
        line_y.grow(modus='from_center', begin_time=t0, transition_time=0)


        # im_ra = RightAngle(location=[5, 0, 0], rotation_euler=[-np.pi / 2, 0, np.pi], color='important', thickness=0.25,
        #                    name='ra_im', radius=0.5)
        # im_ra.appear(begin_time=t0 + 0.5, transition_time=0.5)

        t0 += 0.5
        im_cyl.disappear(begin_time=t0, transition_time=0)
        z_sphere.disappear(begin_time=t0, transition_time=0)
        real_cyl.disappear(begin_time=t0, transition_time=0)

        resolution = 1
        start_frame = int(t0*FRAME_RATE)
        end_frame = int((t0+duration)*FRAME_RATE)
        dt = 1/(end_frame-start_frame)
        for frame in range(start_frame,end_frame,resolution):
            t = (frame+resolution-start_frame)*dt
            pos = mapping(2*np.pi*t)
            f_sphere.ref_obj.location=pos
            # TODO hard coding the motion this should be more convenient later
            # the function move_to cannot be used, since it needs resolution of 2
            ibpy.insert_keyframe(f_sphere.ref_obj,'location',frame+resolution)

        number.update_value(
            lambda frm: coords.location2coords(ibpy.get_location_at_frame(f_sphere,frm))[0],
            begin_time=t0, transition_time=duration)
        number2.update_value(
            lambda frm: coords.location2coords(ibpy.get_location_at_frame(f_sphere,frm))[1],
            begin_time=t0, transition_time=duration)

        # grow projection lines

        sphere.update_position(lambda frm:self.project_x(ibpy.get_location_at_frame(f_sphere,frm)), begin_time=t0,
                               transition_time=duration, resolution=2)
        i_sphere.update_position(lambda frm: self.project_y(ibpy.get_location_at_frame(f_sphere,frm)), begin_time=t0,
                                 transition_time=duration, resolution=2)

        line_x.update_rotation_free_motion(start =lambda frm: self.project_y(ibpy.get_location_at_frame(f_sphere,frm)),
                                           end = lambda frm: ibpy.get_location_at_frame(f_sphere,frm),
                                           begin_time=t0,transition_time=duration,resolution=10)
        line_x2.update_rotation_free_motion(start=lambda frm: Vector(),
                                           end=lambda frm: self.project_x(ibpy.get_location_at_frame(f_sphere, frm)),
                                           begin_time=t0, transition_time=duration, resolution=10)
        line_y.update_rotation_free_motion(start=lambda frm: self.project_x(ibpy.get_location_at_frame(f_sphere, frm)),
                                           end=lambda frm: ibpy.get_location_at_frame(f_sphere, frm),
                                           begin_time=t0, transition_time=duration, resolution=20)
        line_y2.update_rotation_free_motion(start=lambda frm: Vector(),end=lambda frm: self.project_y(ibpy.get_location_at_frame(f_sphere, frm)),
                                           begin_time=t0, transition_time=duration, resolution=20)
        # re_ra.update_position(lambda frm: self.project_y(ibpy.get_location_at_frame(f_sphere,frm)), begin_time=t0,
        #                        transition_time=duration, resolution=2)
        # im_ra.update_position(lambda frm: self.project_x(ibpy.get_location_at_frame(f_sphere, frm)), begin_time=t0,
        #                       transition_time=duration, resolution=2)

    def project_y(self,vector):
        vector = to_vector(vector)
        return Vector([0,vector.y,vector.z])

    def project_x(self,vector):
        vector=to_vector(vector)
        return Vector([vector.x,vector.y,0])

    def svg(self):
        cues = self.sub_scenes['svg']
        t0 = 0  # cues['start']

        sphere = Sphere(0.25)
        sphere.grow(begin_time=t0)
        rope = Rope(start=[0, 0, 0], end=[0, 0.2, 0], length=10, thickness=0.25, resolution=5)
        rope.appear(begin_time=t0)

    def norm_product(self):
        cues = self.sub_scenes['norm_product']
        t0 = 0  # cues['start']

        display = Display(scales=[4, 1], location=[0, 0, 5], number_of_lines=2, flat=True, shadow=False)
        display.appear(begin_time=t0)
        title = SimpleTexBObject(r"|z_1|\cdot|z_2|=|z_1\cdot z_2|", color='important', aligned='center')
        display.set_title(title, shift=[-1, 0])
        t0 += 1
        title.write(begin_time=t0, transition_time=2)
        t0 += 2.5

        display_left = Display(scales=[4, 3], location=[-5, 0, 0], number_of_lines=6, flat=-1)
        display_left.appear(begin_time=t0)
        t0 += 1

        lines = [
            SimpleTexBObject(r"|a+bi|\cdot |c+di|", color='text'),
            SimpleTexBObject(r"\sqrt{a^2+b^2}\cdot \sqrt{c^2+d^2}", color='text'),
            SimpleTexBObject(r"\sqrt{(a^2+b^2)\cdot(c^2+d^2)}", color='text'),
            SimpleTexBObject(r"\sqrt{a^2c^2+a^2d^2+b^2c^2+b^2d^2}", color='text'),
        ]

        for i, line in enumerate(lines):
            if i < 3:
                scale = 0.7
            else:
                scale = 0.6
            display_left.add_text_in(line, line=1.5 * i - 1, indent=0.5, scale=scale)
            line.write(begin_time=t0)
            t0 += 1.5

        display_right = Display(scales=[4, 3], location=[5, 0, 0], number_of_lines=6, flat=False)
        display_right.appear(begin_time=t0)
        t0 += 1

        lines = [
            SimpleTexBObject(r"|(ac-bd)+(ad+bc)i|", color='text'),
            SimpleTexBObject(r"\sqrt{(ac-bd)^2+(ad+bc)^2}", color='text'),
            SimpleTexBObject(r"\sqrt{a^2c^2-2abcd+b^2d^2+a^2d^2+2abcd+b^2c^2}", color='text'),
            SimpleTexBObject(r"\sqrt{a^2c^2+a^2d^2+b^2c^2+b^2d^2}", color='text'),
        ]

        for i, line in enumerate(lines):
            if i != 2:
                scale = 0.6
            else:
                scale = 0.35
            display_right.add_text_in(line, line=1.5 * i - 1, indent=0.5, scale=scale)
            line.write(begin_time=t0)
            t0 += 1.5

        display2 = Display(scales=[7, 1], location=[0, 0, -5], number_of_lines=3, flat=True)
        display2.appear(begin_time=t0)
        title2 = SimpleTexBObject(r"\Longrightarrow |a+bi|\cdot |c+di|=|(ac-bd)+(ad+bc)i|", color='important',
                                  aligned='center')
        display2.set_title(title2, shift=[-2, 0])
        t0 += 1
        title2.write(begin_time=t0, transition_time=3)
        t0 += 3.5
        print(t0)

    def lines(self):
        cues = self.sub_scenes['lines']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, location=[12.5, 0, 0], name='Display')
        title = SimpleTexBObject(r"\text{Lines}", aligned="center",color='example', name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)

        details = 3
        coords = CoordinateSystem(dim=2, lengths=[10 * 7 / 6, 10], domains=[[-2, 5], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[7, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 5.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[-2, 0, -0.5],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)
        # draw grid lines
        coords.draw_grid_lines(colors=['drawing', 'drawing'], begin_time=t0, transition_time=2, sub_grid=5,
                               loop_cuts=details)

        # first line
        t0 += 3
        removables = flatten([coords.x_lines, coords.y_lines])

        zb = 1 + 2j
        za = -1 + 1j

        colors = flatten([['drawing']])

        lines = [
            SimpleTexBObject(
                r"A: " + z2str(za),
                color='drawing'),
            SimpleTexBObject(r"B: " + z2str(zb), color='drawing'),
        ]
        lines[1].align(lines[0], char_index=1, other_char_index=1)

        spheres = [
            Sphere(0.25, location=coords.coords2location(z2p(za)), color='drawing', name=z2str(za)),
            Sphere(0.25, location=coords.coords2location(z2p(zb)), color='drawing', name=z2str(zb)),
            Sphere(0.25, location=coords.coords2location(z2p(za - za)), color='example', name=z2str(za - za)),
            Sphere(0.25, location=coords.coords2location(z2p(zb - za)), color='example', name=z2str(zb - za)),
        ]

        coords.add_objects(spheres)

        for i, line in enumerate(lines):
            display.add_text_in(line, line=i + 1, indent=1.5)
            line.write(begin_time=t0)
            spheres[i].grow(begin_time=t0)
            spheres[i].write_name_as_label(modus='up_left', offset=[-0.2, 0, 0], begin_time=t0 + 0.5,
                                           transition_time=0.5)
            t0 += 1.5

        arrows = [
            PArrow(
                start=coords.coords2location(z2p(za)),
                end=coords.coords2location(z2p(za - za)),
                color='example',
                name='Arrow1'
            ),
            PArrow(
                start=coords.coords2location(z2p(zb)),
                end=coords.coords2location(z2p(zb - za)),
                color='example',
                name='Arrow2'
            ),
            PArrow(
                start=coords.coords2location(z2p(0)),
                end=coords.coords2location(z2p(zb - za)),
                color='example',
                name='Arrow3'
            ),
        ]

        coords.add_objects(arrows)
        removables = flatten([removables, arrows, [spheres[2], spheres[3]]])

        for i in range(0, 2):
            arrow = arrows[i]
            arrow.grow(begin_time=t0)
            spheres[i + 2].grow(begin_time=t0 + 0.5)
            spheres[i + 2].write_name_as_label(modus='up', begin_time=t0 + 1, transition_time=0.5, aligned='left',
                                               offset=[0.1, 0, 0])
        t0 += 2

        lines.append(
            SimpleTexBObject(r"AB:\,\,\lambda\cdot(" + z2str(zb - za) + ")+(" + z2str(za) + ")", color='example')
        )

        arrow_C = arrows[2]
        arrows[0].disappear(begin_time=t0, transition_time=0.5)
        arrows[1].disappear(begin_time=t0, transition_time=0.5)
        arrow_C.grow(begin_time=t0, transition_time=0.5)
        t0 += 1

        display.add_text_in(lines[-1], line=4, indent=1)
        lines[-1].write(letter_range=[0, 10], begin_time=t0)
        t0 += 1

        arrow_C.rescale(rescale=[-1, -1, -1], begin_time=t0, transition_time=0.1)
        t0 += 0.11

        count = 0
        duration = 3
        sphere_t = Sphere(0.25, location=coords.coords2location(z2p(za - zb)), color='example')
        coords.add_object(sphere_t)
        sphere_t.grow(begin_time=t0 + 0.1, transition_time=0.1)

        labels = [
            r"\lambda=-1",
            r"\lambda=-{1\over 2}",
            r"\lambda=0",
            r"\lambda={1\over 2}",
            r"\lambda=1",
            r"\lambda={3\over 2}",
            r"\lambda=2",
        ]
        t0 += 0.11
        arrow_C.rescale(rescale=[-1, -1, -2], begin_time=t0, transition_time=duration)
        sphere_t.move_to(target_location=coords.coords2location(z2p(2.01 * (zb - za))), begin_time=t0,
                         transition_time=duration)

        l_old = np.Infinity
        movers = []
        for frame in range(int(t0 * FRAME_RATE), int((t0 + duration) * FRAME_RATE) + 1):
            location = coords.ref_obj.matrix_world @ ibpy.get_location_at_frame(sphere_t, frame)
            z = p2z(coords.location2coords(location))
            l = np.floor(2 * np.sign(np.real(z)) * np.abs(z) / np.abs(zb - za)) / 2
            if l_old != l:
                l_old = l
                sphere = Sphere(0.25, location=coords.coords2location(z2p(l * (zb - za))), name=labels[count],
                                color='example', emission=0)
                t = frame / FRAME_RATE
                sphere.grow(begin_time=t, transition_time=0.1)
                count += 1
                sphere.write_name_as_label(modus='down_right', begin_time=t + 0.1, transition_time=0.2)
                coords.add_object(sphere)
                movers.append(sphere)

        t0 += duration + 1.5

        removables.append(sphere_t)
        explosion = Explosion(removables)
        explosion.set_wind_and_turbulence(wind_location=[6, 0, -5], turbulence_location=[0, 0, 0],
                                          rotation_euler=[0, -np.pi / 4, 0], wind_strength=1.5, turbulence_strength=10)
        for i in range(2, 4):
            spheres[i].label_disappear(begin_time=t0, transition_time=0.5)
        explosion.explode(begin_time=t0, transition_time=2)

        t0 += 2.5

        line = Cylinder.from_start_to_end(
            start=coords.coords2location(z2p(-1.5 * (zb - za))),
            end=coords.coords2location(z2p(2.5 * (zb - za))),
            color='example',
            thickness=0.5, emission=0,
        )
        coords.add_object(line)
        movers.append(line)
        line.grow(modus='from_start', begin_time=t0)

        t0 += 1.5

        lines[-1].write(letter_range=[10, len(lines[-1].letters)], begin_time=t0)
        t0 += 1.5

        arrow_D = PArrow(end=coords.coords2location(z2p(za)), color='example')
        coords.add_object(arrow_D)
        arrow_D.grow(begin_time=t0)

        for mover in movers:
            mover.move(direction=coords.coords2location(z2p(za)), begin_time=t0)
        t0 += 1.5

        for mover in movers:
            mover.change_color(new_color='drawing', begin_time=t0, transition_time=0.5)
            if isinstance(mover, Sphere):
                mover.disappear(begin_time=t0 + 0.5, transition_time=0.5)

        arrow_D.disappear(t0 + 0.5, transition_time=0.5)

        lines.append(
            SimpleTexBObject(r"AB:\,\,(2\lambda-1)+(\lambda+1)i", color='drawing')
        )

        display.add_text_in(lines[-1], line=5, indent=1)
        src = lines[-2]
        target = lines[-1]

        target.write(letter_range=[0, 4], begin_time=t0, transition_time=0.3)
        t0 += 0.3
        src.move_copy_to(target, src_letter_indices=[6, 3, 12, 13], target_letter_indices=[4, 5, 6, 7], begin_time=t0,
                         transition_time=0.3, new_color='drawing')
        t0 += 0.3
        target.write(letter_range=[8, 11], begin_time=t0, transition_time=0.3)
        t0 += 0.3
        src.move_copy_to(target, src_letter_indices=[3, 8, 15], target_letter_indices=[11, 15, 15], begin_time=t0,
                         transition_time=0.3, new_color='drawing')
        target.write(letter_range=[12, 15], begin_time=t0, transition_time=0.3)
        t0 += 0.5

        lines = [
            SimpleTexBObject(r"\text{Result:} ", color='text'),
            SimpleTexBObject(r"AB:\,\, z_1+\lambda(z_2-z_1)", color='example')
        ]

        for i, line in enumerate(lines):
            if i == 0:
                indent = 0.5
            else:
                indent = 1
            display.add_text_in(line, line=i + 7, indent=indent)

        lines[-2].write(begin_time=t0, transition_time=0.5)
        lines[-1].write(begin_time=t0 + 0.6)
        t0 += 2

        # second line

        zc = 4 + 1j
        zd = 5 - 1j

        colors = flatten([['important']])

        lines = [
            SimpleTexBObject(r"C: z_1=" + z2str(zc),color=colors),
            SimpleTexBObject(r"D: z_2=" + z2str(zd), color=colors),
        ]
        lines[1].align(lines[0], char_index=7, other_char_index=7)

        spheres = [
            Sphere(0.25, location=coords.coords2location(z2p(zc)), color='important', name=z2str(zc)),
            Sphere(0.25, location=coords.coords2location(z2p(zd)), color='important', name=z2str(zd)),
        ]

        coords.add_objects(spheres)

        for i, line in enumerate(lines):
            display.add_text_in(line, line=i + 11, indent=1.5)
            line.write(begin_time=t0)
            spheres[i].grow(begin_time=t0)
            spheres[i].write_name_as_label(modus='left', offset=[-0.3, 0, 0], begin_time=t0 + 0.5,
                                           transition_time=0.5, aligned='right')
            t0 += 1.5

        lines = [
            SimpleTexBObject(r"CD:\,\, z_1+\mu(z_2-z_1)", color='example'),
            SimpleTexBObject(r"CD:\,\, (4+i)+\mu\left(\rule{0em}{2ex}(5-i)-(4+i)\right)", color='text'),
            SimpleTexBObject(r"CD:\,\, (4+i)+\mu\,\,(\,\,1-2i)", color='text'),
            SimpleTexBObject(r"CD:\,\, (\mu+4)+(-2\mu+1)i", color='text')
        ]

        algebra_indent = 1
        for i, line in enumerate(lines):
            display.add_text_in(line, line=i + 14, indent=algebra_indent)
            if i < 2:
                line.write(begin_time=t0)
                t0 += 1.5

        indices = [11, 15, 17, 21]
        for i in indices:
            lines[1].letters[i].disappear(begin_time=t0, transition_time=0.5)

        replacement = SimpleTexBObject(r"CD:\,\,(4+i)+\mu\left(\rule{0em}{3ex}(5-i)-(4-i)\right)", color='text')
        lines[1].replace(replacement, src_letter_range=[19, 20], img_letter_range=[19, 20], begin_time=t0,
                         transition_time=0.5)
        t0 += 1
        lines[2].write(letter_range=[0, 11], begin_time=t0, transition_time=1)
        t0 += 1.1
        indices = [12, 16, 18]
        for i in indices:
            lines[1].letters[i].change_color(new_color='joker', begin_time=t0, transition_time=0.5)
        lines[2].write(letter_range=[11, 12], begin_time=t0 + 0.5, transition_time=0.5)
        lines[2].letters[11].change_color(new_color='joker', begin_time=t0, transition_time=0.5)
        t0 += 1.5

        indices = [13, 14, 19, 20]
        for i in indices:
            lines[1].letters[i].change_color(new_color='important', begin_time=t0, transition_time=0.5)
        lines[2].write(letter_range=[12, 15], begin_time=t0 + 0.5, transition_time=0.5)
        lines[2].letters[12].change_color(new_color='important', begin_time=t0, transition_time=0.5)
        lines[2].letters[13].change_color(new_color='important', begin_time=t0, transition_time=0.5)
        lines[2].letters[14].change_color(new_color='important', begin_time=t0, transition_time=0.5)
        t0 += 1
        lines[2].write(letter_set=[15], begin_time=t0, transition_time=0.1)
        t0 += 0.5
        lines[2].letters[11].change_color(new_color='text', begin_time=t0, transition_time=0.5)
        lines[2].letters[12].change_color(new_color='text', begin_time=t0, transition_time=0.5)
        lines[2].letters[13].change_color(new_color='text', begin_time=t0, transition_time=0.5)
        lines[2].letters[14].change_color(new_color='text', begin_time=t0, transition_time=0.5)

        lines[3].write(letter_range=[0, 3], begin_time=t0, transition_time=0.5)
        lines[3].write(letter_set=[3, 5, 7, 8, 9, 13, 15], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5

        lines[2].move_copy_to(lines[3], src_letter_indices=[4, 9], target_letter_indices=[4, 6], begin_time=t0)
        t0 += 1.5
        lines[2].move_copy_to_and_remove(lines[3], src_letter_indices=[6, 9, 12, 13,14], target_letter_indices=[16, 12, 10, 11,16],
                              begin_time=t0)
        lines[3].write(letter_set=[14], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 2

        count = 0
        duration = 3
        sphere_t = Sphere(0.25, location=coords.coords2location(z2p(-1.5 * (zd - zc) + zc)), color='important')
        l2 = Cylinder.from_start_to_end(
            start=coords.coords2location(z2p(-1.5 * (zd - zc) + zc)),
            end=coords.coords2location(z2p(2 * (zd - zc) + zc)),
            color='important',
            thickness=0.5
        )
        coords.add_objects([sphere_t, l2])
        sphere_t.grow(begin_time=t0 + 0.1, transition_time=0.1)

        mus = [
            r"\mu=-{3\over 2}",
            r"\mu=-1",
            r"\mu=-{1\over 2}",
            r"\mu=0",
            r"\mu={1\over 2}",
            r"\mu=1",
            r"\mu={3\over 2}",
        ]
        t0 += 0.11
        sphere_t.move_to(target_location=coords.coords2location(z2p(1.51 * (zd - zc) + zc)), begin_time=t0,
                         transition_time=duration)
        l2.grow(modus='from_start', begin_time=t0)

        l_old = np.Infinity
        movers = [sphere_t]
        for frame in range(int(t0 * FRAME_RATE), int((t0 + duration) * FRAME_RATE) + 1):
            location = coords.ref_obj.matrix_world @ ibpy.get_location_at_frame(sphere_t, frame)
            z = p2z(coords.location2coords(location)) - zc
            l = np.floor(2 * np.sign(np.real(z)) * np.abs(z) / np.abs(zd - zc)) / 2
            if l_old != l:
                l_old = l
                sphere = Sphere(0.25, location=coords.coords2location(z2p(l * (zd - zc) + zc)), name=mus[count],
                                color='important', emission=0)
                t = frame / FRAME_RATE
                sphere.grow(begin_time=t, transition_time=0.1)

                if count == 4:
                    modus = 'up_right'
                    aligned = 'left'
                    offset = [0, 0, 0]
                elif count < 6:
                    modus = 'right'
                    offset = [0.4, 0, 0]
                    aligned = 'left'
                else:
                    modus = 'down'
                    offset = [0, -0.2, 0]
                    aligned = 'center'

                sphere.write_name_as_label(modus=modus, begin_time=t + 0.1, offset=offset, transition_time=0.2,
                                           aligned=aligned)
                coords.add_object(sphere)
                movers.append(sphere)
                count += 1

        t0 += duration + 1.5

        for mover in movers:
            mover.disappear(begin_time=t0)
            t0 += 0.1

        t0 += 2

        title = SimpleTexBObject(r"\text{The intersection of lines}", color='example', aligned='center')
        display.set_title_back(title)
        title.write(begin_time=t0)

        display.rotate(rotation_euler=[np.pi / 2, 0, -np.pi - np.pi / 15], begin_time=t0)
        sphere_S = Sphere(0.25, location=coords.coords2location(z2p(-(zd - zc) + zc)), color='example',
                          name=r'z\leftrightarrow(S)')
        coords.add_object(sphere_S)
        sphere_S.grow(begin_time=t0)

        t0 += 2

        coords.rotate(rotation_euler=[-np.pi / 3, 0, 0], begin_time=t0, transition_time=2)
        coords.move(direction=[0, 0, -2], begin_time=t0, transition_time=2)
        t0 += 2.5

        l2.change_color(new_color='important', begin_time=t0)
        t0 += 1.1

        lines = [
            SimpleTexBObject(r"AB:\,\,(2\lambda-1)+(\lambda+1)i", color='drawing'),
            SimpleTexBObject(r"CD:\,\,(\mu+4)+(-2\mu+1)i", color='important')
        ]

        duration = 5
        for i, line in enumerate(lines):
            display.add_text_in_back(line, line=i + 1, indent=1)
            line.write(begin_time=t0 + i * duration + 0.5, transition_time=1)

        # first flag moving on line
        spheres = [
            Sphere(0.25, location=coords.coords2location(z2p(line1(-1.5))), color='drawing'),
            Sphere(0.25, location=coords.coords2location(z2p(line2(2))), color='important')
        ]
        coords.add_objects(spheres)
        spheres[0].grow(begin_time=t0, transition_time=0.5)
        spheres[1].grow(begin_time=t0 + duration, transition_time=0.5)

        spheres[0].move_to(target_location=z2loc(coords, line1(2.5)), begin_time=t0, transition_time=duration * 0.8)
        spheres[0].move_to(target_location=z2loc(coords, line1(2)), begin_time=t0 + duration * 0.8,
                           transition_time=duration * 0.2)

        spheres[1].move_to(target_location=z2loc(coords, line2(-1.5)), begin_time=t0 + duration + 0.5,
                           transition_time=duration * 0.8)
        spheres[1].move_to(target_location=z2loc(coords, line2(-1)), begin_time=t0 + duration * 1.8,
                           transition_time=duration * 0.2)

        # show flags and change the image on the flags
        flags = [
            Flag(colors=['text', 'drawing'], name='Flag1', rotation_euler=[np.pi / 3, 0, 0],
                 location=z2loc(coords, line1(-1.5)), simulation_start=t0,
                 simulation_duration=3 * duration),
            Flag(colors=['text', 'important'], name='Flag2', mirror=True, rotation_euler=[-np.pi / 3, 0, np.pi],
                 location=z2loc(coords, line2(1.5)),
                 simulation_start=t0 + duration,
                 simulation_duration=3 * duration)
        ]

        coords.add_objects(flags)
        flags[0].appear(begin_time=t0, transition_time=0.1)
        flags[1].appear(begin_time=t0 + duration, transition_time=0.5)

        l1_old = -np.Infinity
        l2_old = -np.Infinity
        count1 = 0
        count2 = 0
        frame1_old = int(t0 * FRAME_RATE)
        frame2_old = int(t0 * FRAME_RATE)
        for frame in range(int(t0 * FRAME_RATE), int((t0 + 2 * duration + 0.5) * FRAME_RATE)):
            loc1 = ibpy.get_location_at_frame(spheres[0], frame)
            loc2 = ibpy.get_location_at_frame(spheres[1], frame)
            m = ibpy.get_world_matrix(coords)
            z1 = p2z(coords.location2coords(m @ loc1)) - za
            z2 = p2z(coords.location2coords(m @ loc2)) - zc
            l1 = np.real(z1 / (zb - za))
            l2 = np.real(z2 / (zd - zc))

            l1 = np.round(l1 * 10) / 10
            l2 = np.round(l2 * 10) / 10

            if l1_old != l1:
                l1_old = l1
                if l1 >= 0:
                    sgn = "+"
                else:
                    sgn = ""
                text = r'\text{\fbox{$\rule{0em}{2ex}\lambda=' + sgn + str(l1) + '$}}'
                ic = ImageCreator(text, count1, prefix=flags[0].name)
                count1 += 1
                image = ic.get_image_path()
                flags[0].add_image_texture(image, frame)
                flags[0].move_to(target_location=z2loc(coords, line1(l1)), begin_time=frame1_old / FRAME_RATE,
                                 transition_time=(frame - frame1_old) / FRAME_RATE)
                frame1_old = frame
            if l2_old != l2:
                l2_old = l2
                if l2 >= 0:
                    sgn = "+"
                else:
                    sgn = ""
                text = r'\text{\fbox{$\rule{0em}{1.8ex}\mu=' + sgn + str(l2) + '$}}'
                ic = ImageCreator(text, count2, prefix=flags[1].name)
                count2 += 1
                image = ic.get_image_path()
                flags[1].add_image_texture(image, frame)
                flags[1].move_to(target_location=z2loc(coords, line2(l2)),
                                 begin_time=frame2_old / FRAME_RATE,
                                 transition_time=(frame - frame2_old) / FRAME_RATE)
                frame2_old = frame
        t0 += 2 * duration

        t0 += 1
        color1 = flatten([['drawing'] * 2, ['text'], ['important'] * 2])
        lines2 = [
            SimpleTexBObject(r"AB=CD", color=color1),
            SimpleTexBObject(r"2\lambda-1=\mu+4", color='text'),
            SimpleTexBObject(r"\lambda+1=-2\mu+1", color='text')
        ]

        display.add_text_in_back(lines2[0], line=4, indent=2)
        display.add_text_in_back(lines2[1], line=6)
        display.add_text_in_back(lines2[2], line=7)

        for i in range(1, 3):
            lines2[i].align(lines2[0], char_index=5 - i, other_char_index=2)

        lines[0].move_copy_to(lines2[0], src_letter_indices=[0, 1], target_letter_indices=[0, 1],
                              begin_time=t0, offset=[1.2, 0, 0])
        t0 += 1.1
        lines2[0].write(letter_set=[2], begin_time=t0, transition_time=0.1)
        lines[1].move_copy_to(lines2[0], src_letter_indices=[0, 1], target_letter_indices=[3,4],
                              begin_time=t0, offset=[1.2, 0, 0])
        t0 += 1.6

        lines[0].move_copy_to(lines2[1], src_letter_indices=[4, 5, 6, 7], target_letter_indices=[0, 1, 2, 3],
                              begin_time=t0, offset=[1.2, 0, 0], new_color='text')
        t0 += 1.1
        lines2[1].write(letter_set=[4], begin_time=t0, transition_time=0.1)
        lines[1].move_copy_to(lines2[1], src_letter_indices=[4, 5, 6], target_letter_indices=[5, 6, 7],
                              begin_time=t0, offset=[1.2, 0, 0], new_color='text')
        t0 += 1.6

        lines[0].move_copy_to(lines2[2], src_letter_indices=[11, 12, 13], target_letter_indices=[0, 1, 2],
                              begin_time=t0, offset=[1.2, 0, 0], new_color='text')
        t0 += 1.1
        lines2[2].write(letter_set=[3], begin_time=t0, transition_time=0.1)
        lines[1].move_copy_to(lines2[2], src_letter_indices=[10, 11, 12, 13, 14], target_letter_indices=[4, 5, 6, 7, 8],
                              begin_time=t0, offset=[1.2, 0, 0], new_color='text')
        t0 += 1.6
        underline = display.add_line(start=[2, 5.5], end=[6.5, 5.5], color='text', thickness=0.0675)
        underline.grow(modus='from_start', begin_time=t0)
        t0 += 1.5

        lines3 = [
            SimpleTexBObject('\lambda=2', color='drawing'),
            SimpleTexBObject('\mu=-1', color='important'),
            SimpleTexBObject(r'z=3+3i', color='example'),
            SimpleTexBObject(r'S=(3,3)', color='example'),
        ]

        for i, line in enumerate(lines3):
            display.add_text_in_back(line, line=9 + i, indent=2)

        lines3[0].write(begin_time=t0)
        lines3[1].write(begin_time=t0)
        t0 += 1.5

        flags[0].disappear(begin_time=t0)
        flags[1].disappear(begin_time=t0)

        t0 += 1.5
        coords.rotate(rotation_euler=[0, 0, 0], begin_time=t0)
        t0 += 1.5
        lines3[2].write(begin_time=t0)
        lines3[3].write(begin_time=t0)
        sphere_S.write_name_as_label(modus='right', aligned='left', offset=[0.4, 0, 0], begin_time=t0)
        t0 += 1.5

        t0 += 1
        print(t0)

    def i2m_one(self):
        """
        post processing required for this scene
        point light turn down from 10000 to 0 within 60s
        force turn down from 5000 to 0 within 10 s

        scattering and absorption turn down to 0 within 100 s

        use QUALITY='exceptional' for nice slow motion (in constants.py)
        render settings in blender
        increase noise-threshold =0.001
        maxSamples=256  for the first 160 frames

        :return:
        """
        cues = self.sub_scenes['i2m_one']
        t0 = 0  # cues['start']

        lines = [
            SimpleTexBObject(r"i^2=-1", aligned='center', text_size='huge', color='gold',
                             thickness=5, bevel=3, name='Treasure', emission=1000)
        ]
        lines[0].write(begin_time=t0, transition_time=0)
        for letter in lines[0].letters:
            change_emission_to(letter, 500, begin_time=t0 + 0.1, transition_time=0.5)
            change_emission_to(letter, 1, begin_time=t0 + 0.6, transition_time=0.5)
            change_emission_to(letter, 0.3, begin_time=t0 + 1.1, transition_time=2)

        ibpy.set_sun_light(energy=0)
        ibpy.set_camera_location(location=[0, -8, -0.5])
        ibpy.set_camera_lens(lens=15)

        ibpy.set_hdri_background('autoshop_01_4k')
        ibpy.set_hdri_strength(0, begin_time=0, transitions_time=0)
        ibpy.set_hdri_strength(1, begin_time=1, transitions_time=.5)

        ibpy.camera_zoom(lens=70, begin_time=2, transition_time=1)

        # area = ibpy.add_area_light(location=[3,-6,0],energy=200)
        # barea = BObject(obj=area)
        # barea.rescale(rescale=[1,2,1],begin_time=t0,transition_time=0)
        # area.rotate(rotation_euler=[0,0,0],begin_time=t0,transition_time=0)
        # ibpy.set_track(area,target=lines[0])
        # ibpy.add_point_light(location=[-1.3,-0.11,0.18],scale=[1,3,1],energy=100)
        # ibpy.add_point_light(location=[1.4,0.11,-0.4],energy=100)
        # ibpy.add_point_light(location=[0,0,0],energy=1000)
        force = Force(name='Force', strength=5000)
        force.disappear(begin_time=0, transition_time=0.5)

        sphere = Sphere(2.25, resolution=5, color='marble', name='MarbleSphere', emission=0.3)
        change_emission_to(sphere, 0, begin_time=1, transition_time=0.5)
        sphere.grow(begin_time=t0, transition_time=0)

        lines[0].rotate(rotation_euler=[np.pi / 2, 0, 5 * 2 * np.pi], begin_time=3, transition_time=50)

        plane = Plane(location=[0, 0, -2.4], scale=[4, 4, 1], color='sand', name='SandPlane', emission=0.5)
        plane.appear(begin_time=t0, transition_time=0)

        sphere_volume = Sphere(10, color='scatter_volume', name='ScatteringSphere')
        sphere_volume.appear(begin_time=t0, transition_time=0)
        ibpy.set_volume_scatter(sphere_volume, 0.0025, begin_time=t0)
        ibpy.change_volume_scatter(sphere_volume, 0, begin_time=t0, transition_time=1)

        ibpy.set_volume_absorption(sphere_volume, 0.0, begin_time=t0)
        ibpy.change_volume_absorption(sphere_volume, 0.01, begin_time=t0 + 0.1, transition_time=1)

    def debug2(self):
        """
        solve problem with tic marks
        :return:
        """
        cues = self.sub_scenes['debug2']
        t0 = 0  # cues['start']

        ra = RightAngle(color='drawing',thickness=0.5,name='RightAngle')
        ra.appear(begin_time=t0)

    def basics2(self):
        cues = self.sub_scenes['basics2']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=8, scales=[4,2],columns=1, location=[-2.5,0, 0], name='Display',flat=True)
        title = SimpleTexBObject(r"\text{Complex Numbers}", aligned="center", name='Title', color='text')
        display.set_title(title,shift=[-0.5,0])

        display.appear(begin_time=t0, transition_time=0)
        title.write(begin_time=t0, transition_time=0)

        colors = flatten([['text']*2,['joker'],['text'],['important']])
        lines = [
            SimpleTexBObject(r'z=a+b\,i',color = colors),
            SimpleTexBObject(r'\text{real part}',color='joker',aligned='center'),
            SimpleTexBObject(r'\text{imaginary part}',color='important',aligned='center'),
        ]

        indents = [1,5,8]
        rows = [3,5,1]
        scales = [2.5,0.7,0.7]
        for i, line in enumerate(lines):
            display.add_text_in(line, line=rows[i], indent=indents[i], scale=scales[i])
            line.write(begin_time=t0,transition_time=0)
        display.rescale(rescale=[0.7,0.7,1],begin_time=t0,transition_time=0)

        display2 = Display(scales=[8,3], number_of_lines=5.5, location=[9, 0, 3], columns=1,flat=True)
        title2 = SimpleTexBObject(r"\text{Addition and Subtraction}", color='text', aligned="center")
        display2.set_title(title2)
        display2.appear(begin_time=t0,transition_time=0)
        title2.write(begin_time=t0,transition_time=0)

        display3 = Display(scales=[8, 3], number_of_lines=5.5, location=[9, 0, -3], columns=1, flat=True)
        title3 = SimpleTexBObject(r"\text{Multiplication and Division}", color='text', aligned="center")
        display3.set_title(title3)

        display3.appear(begin_time=t0)
        title3.write(begin_time=t0, transition_time=0)

        t0 += 0.5

        # add
        color5 = flatten(
            [['text'], ['joker'], ['text'], ['important'] * 2, ['text'] * 3, ['joker'], ['text'], ['important'] * 2,['text']*2,
             ['joker'] * 5, ['text'], ['important'] * 6])

        line5 = SimpleTexBObject(r"(a+b\,i)+(c+d\,i)=(a+c)+(b+d)\,i", color=color5,name='line5',bevel=0)

        indent = 0.65
        display2.add_text_in(line5, line=1, indent=indent, column=1)
        line5.write(letter_range=[0, 14], begin_time=t0)
        t0 += 1.5

        # sub
        color6 = flatten(
            [['text'], ['joker'], ['text'], ['important'] * 2, ['text'] * 3, ['joker'], ['text'], ['important'] * 2,
             ['text'] * 2,
             ['joker'] * 5, ['text'], ['important'] * 6])

        line6 = SimpleTexBObject(r"(a+b\,i)-(c+d\,i)=(a-c)+(b-d)\,i", color=color6, name='line6',bevel=0)

        indent = 0.65
        display2.add_text_in(line6, line=2, indent=indent, column=1)
        line6.write(letter_range=[0, 14], begin_time=t0)
        t0 += 1.5

        # mul and div
        color7 = flatten(
            [['text'], ['joker'], ['text'], ['important'] * 2, ['text'] * 3, ['joker'], ['text'], ['important'] * 2,
             ['text'] * 2, ['joker'] * 2, ['text'], ['important'] * 5, ['text'], ['important'] * 3, ['text'],
             ['important']*3])
        color8 =  flatten(
            [['text'], ['joker'], ['text'], ['important'] * 2, ['text'] * 3, ['joker'], ['text'], ['important'] * 2,
             ['text'] * 2, ['joker'] * 2, ['text'], ['joker'] * 7, ['text'], ['important'] * 3, ['text'],
             ['important'] * 3])
        color9 = flatten(
            [['text'], ['joker'], ['text'], ['important'] * 2, ['text'] * 3, ['joker'], ['text'], ['important'] * 2,
             ['text'] * 2,  ['joker'] * 7, ['text'], ['important'] * 8])
        color10 = flatten(
            [['text']*2, ['joker']*2, ['text']*3, ['important'] * 4, ['text'] * 3, ['joker']*13, ['text'], ['important'] *14 ])

        line7 = SimpleTexBObject(r"(a+b\,i)\cdot (c+d\,i)=ac+\,\,i^2\hspace{0.26em}\,\,\,\,\cdot bd+ad\,i+bc\,i",
                                 color=color7,name='line7',bevel=0)
        line8 = SimpleTexBObject(r"(a+b\,i)\cdot (c+d\,i)=ac+(-1)\cdot bd+ad\,i+bc\,i",
                                 color=color8, name='line8',bevel=0)
        line9 = SimpleTexBObject(r"(a+b\,i)\cdot (c+d\,i)=(ac-bd)+(ad+bc)\,i",
                                 color=color9, name='line9',bevel=0)
        line10 = SimpleTexBObject(r"{(a+b\,i)\over (c+d\,i)}=\left({ac+bd\over c^2+d^2}\right)+\left({bc-ad\over c^2+d^2}\right)i" ,color=color10, name='line10', bevel=0)

        indent=0.25
        display3.add_text_in(line7, line=0.5, indent=indent, column=1)
        display3.add_text_in(line8, line=0.55, indent=indent, column=1)
        display3.add_text_in(line9, line=0.5, indent=indent, column=1)
        display3.add_text_in(line10, line=2.5, indent=indent, column=1)
        line7.write(letter_range=[0, 14], begin_time=t0)
        t0 += 1.5
        line10.write(letter_set=[0,2,4,7,9,11],begin_time=t0,transition_time=0.5)
        t0+=0.5
        line10.write(letter_set=[6],begin_time=t0,transition_time=0.1)
        t0+=0.1
        line10.write(letter_set=[1,3,5,8,10,12], begin_time=t0, transition_time=0.5)
        t0 += 0.5
        line10.write(letter_set=[13], begin_time=t0, transition_time=0.1)
        t0 += 1

        display4 = Display(number_of_lines=8, scales=[0.7 * 4, 0.7 * 2], columns=1, location=[-2.5, 0, -3],
                           name='DisplayCC', flat=True)
        title4 = SimpleTexBObject(r"\text{For division:}", aligned="center", name='Title', color='text')
        display4.set_title(title4, shift=[-0.5, 0])

        display4.appear(begin_time=t0, transition_time=0.5)
        title4.write(begin_time=t0, transition_time=0.5)

        colors = flatten([['joker'] , ['text'], ['important']*2,['text']*2])
        lines = [
            SimpleTexBObject(r'c+d\,i\neq 0', color=colors),
        ]

        t0 += 0.5
        indents = [1]
        rows = [2]
        scales = [2 * 0.7]
        for i, line in enumerate(lines):
            display4.add_text_in(line, line=rows[i], indent=indents[i], scale=scales[i])
            line.write(begin_time=t0, transition_time=0.5)
        t0+=1

        # solutions
        # add
        line5.move_copy_to(src_letter_indices=[1], target_letter_indices=[15], begin_time=t0,
                           transition_time=0.5,detour=display3.up)
        t0+=0.5
        line5.write(letter_set=[16], begin_time=t0, transition_time=0.25)
        t0+=0.25
        line5.move_copy_to(src_letter_indices=[8], target_letter_indices=[18], begin_time=t0,
                           transition_time=0.5,detour=display3.up)
        t0+=0.5
        line5.write(letter_set=[14, 17, 19], begin_time=t0, transition_time=0.5)
        t0 += 1

        line5.move_copy_to(src_letter_indices=[3,4], target_letter_indices=[21,25], begin_time=t0,
                           transition_time=0.5,detour=display3.up)
        t0+=0.5
        line5.write(letter_set=[22], begin_time=t0, transition_time=0.25)
        t0 += 0.25
        line5.move_copy_to(src_letter_indices=[10, 11], target_letter_indices=[ 23, 25], begin_time=t0,
                           transition_time=0.5,detour=display3.up)
        t0+=0.5
        line5.write(letter_set=[20, 24], begin_time=t0, transition_time=0.25)
        t0+=1

        # sub
        line6.move_copy_to(src_letter_indices=[1, 8], target_letter_indices=[15, 18], begin_time=t0,
                           transition_time=0.5,detour=display3.down)
        line6.move_copy_to(src_letter_indices=[6], target_letter_indices=[16], begin_time=t0,
                           transition_time=0.5,new_color='joker',detour=display3.down)

        line6.write(letter_set=[14, 16, 17, 19], begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5
        line6.move_copy_to(src_letter_indices=[3, 4, 10, 11], target_letter_indices=[21, 25, 23, 25], begin_time=t0,
                           transition_time=0.5,detour=display3.down)
        line6.move_copy_to(src_letter_indices=[6], target_letter_indices=[22], begin_time=t0,
                           transition_time=0.5, new_color='important',detour=display3.down)
        line6.write(letter_set=[20, 24], begin_time=t0, transition_time=0.5)
        t0 += 1.5

        # mul
        line7.move_copy_to_and_remove(src_letter_indices=[1, 8], target_letter_indices=[14, 15], begin_time=t0,
                           transition_time=0.5,detour=display3.down)
        t0+=1
        line7.write(letter_set=[16],begin_time=t0,transition_time=0)
        line7.move_copy_to_and_remove(src_letter_indices=[3,4],target_letter_indices=[20,17],begin_time=t0,transition_time=0.5,detour=display3.up)
        line7.move_copy_to_and_remove(src_letter_indices=[10,11],target_letter_indices=[21,17],begin_time=t0,transition_time=0.5,detour=display3.down)
        t0+=0.5
        line7.write(letter_set=[18,19],begin_time=t0,transition_time=0)
        t0+=0.5
        line7.write(letter_set=[22], begin_time=t0, transition_time=0)
        line7.move_copy_to_and_remove(src_letter_indices=[1], target_letter_indices=[23], begin_time=t0, transition_time=0.5, detour=display3.down,new_color='important')
        line7.move_copy_to_and_remove(src_letter_indices=[10,11], target_letter_indices=[24,25], begin_time=t0,
                                      transition_time=0.5, detour=display3.down)
        t0+=1
        line7.write(letter_set=[26], begin_time=t0, transition_time=0)
        line7.move_copy_to_and_remove(src_letter_indices=[8], target_letter_indices=[28], begin_time=t0,
                                      transition_time=0.5, detour=display3.up, new_color='important')
        line7.move_copy_to_and_remove(src_letter_indices=[3, 4], target_letter_indices=[27, 29], begin_time=t0,
                                      transition_time=0.5, detour=display3.down)
        t0+=1
        line7.replace(line8,src_letter_range=[17,19],img_letter_range=[17,21],begin_time=t0,transition_time=0.5,morphing=False)
        set = range(19,22)
        for s in set:
            line7.letters[s].change_color(new_color='joker',begin_time=t0,transition_time=0.5)
        line7.perform_morphing()
        t0+=0.5
        line8.write(begin_time=t0,transition_time=0)
        line7.disappear(begin_time=t0,transition_time=0.5)
        t0+=1
        line8.replace(line9,src_letter_range=[14,24],img_letter_range=[14,21],begin_time=t0,transition_time=1,morphing=False)
        line8.replace(line9,src_letter_range=[24,32],img_letter_range=[21,30],begin_time=t0,transition_time=1,morphing=False)
        line8.perform_morphing()
        for i,n in enumerate(line8.created_null_curves):
            if i>2:
                n.change_color(new_color='important',begin_time=t0)

        t0+=1.5
        line10.write(letter_set=[14,20,26],begin_time=t0,transition_time=0.3)
        t0+=0.3
        line10.write(letter_set=[15,17,19,22,24], begin_time=t0, transition_time=0.5)
        t0+=0.5
        line10.write(letter_set=[16,18,21,23,25], begin_time=t0, transition_time=0.5)
        t0+=1
        line10.write(letter_set=[27,28,34,40,41], begin_time=t0, transition_time=0.5)
        t0 += 0.5
        line10.write(letter_set=[29,31,33,36,38], begin_time=t0, transition_time=0.5)
        t0 += 0.5
        line10.write(letter_set=[30,32,35,37,39], begin_time=t0, transition_time=0.5)
        t0 += 1.5
        print('last frame ', t0 * FRAME_RATE)

    def basics(self):
        cues = self.sub_scenes['basics']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=8, scales=[4, 2], columns=1, location=[12.5, 0, 3.333], name='Display')
        title = SimpleTexBObject(r"\text{Complex Numbers}", aligned="center", name='Title', color='text')
        display.set_title(title,shift=[-0.5,0])

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0+0.5, transition_time=1)

        colors = flatten([['text'] * 2, ['joker'], ['text'], ['important']])
        lines = [
            SimpleTexBObject(r'z=a+b\,i', color=colors),
            SimpleTexBObject(r'\text{real part}', color='joker', aligned='center'),
            SimpleTexBObject(r'\text{imaginary part}', color='important', aligned='center'),
        ]

        t0 += 2

        indents = [1, 5, 8]
        rows = [3, 5, 1]
        scales = [2.5, 0.7, 0.7]
        for i, line in enumerate(lines):
            display.add_text_in(line, line=rows[i], indent=indents[i], scale=scales[i])
            line.write(begin_time=t0, transition_time=1)
            t0 += 1.5

        display.move_to(target_location=[-2.5, 0, 0], begin_time=t0, transition_time=1)
        display.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=1)
        display.rescale(rescale=[0.7, 0.7, 1], begin_time=t0, transition_time=1)

        t0 += 2

        display2 = Display(scales=[8, 3], number_of_lines=5.5, location=[9, 0, 3], columns=1, flat=True)
        title = SimpleTexBObject(r"\text{Addition and Subtraction}", color='text', aligned="center")
        display2.set_title(title)

        display2.appear(begin_time=t0,transition_time=1)
        t0 += 1.5
        title.write(begin_time=t0, transition_time=1)
        t0 += 1.25

        z1 = 3 - 1j
        z2 = -1 + 1j

        colors0 = flatten(
            [['text'] * 3, ['joker'], ['important'] * 2, ['text'] * 3, ['joker'] * 2, ['text'], ['important']])

        line0 = SimpleTexBObject(r'z_1=' + z2str(z1) + '\,\,\,\,\,\,z_2=' + z2str(z2), color=colors0,bevel=0)
        display2.add_text_in(line0, line=0.5, indent=1)

        line0.write(letter_set=[0,1,2,3 ], begin_time=t0, transition_time=.3)
        t0+=0.25
        line0.write(letter_set=[4,5],begin_time=t0,transition_time=0.15)
        t0+=0.2
        line0.write(letter_set=[6,7,8,9,10],begin_time=t0,transition_time=0.3)
        t0+=0.4
        line0.write(letter_set=[11],begin_time=t0,transition_time=0.1)
        t0+=0.2
        line0.write(letter_set=[12],begin_time=t0,transition_time=0.1)
        t0+=0.2

        #addition

        color1 = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3,
             ['joker'] * 2, ['text'], ['important'],
             ['text'] * 2, ['joker'] * 3, ['important'] * 4, ['text'], ['joker']])

        line1 = SimpleTexBObject(r"(3-i)+(-1+i)=3-1-i+i=2", color=color1, bevel=0)
        indent = 0.25
        display2.add_text_in(line1, line=1.5, indent=indent, column=1)

        line0.move_copy_to_and_remove(line1,src_letter_indices=[3,4,5],target_letter_indices=[1,2,3],begin_time=t0,transition_time=0.7)
        line1.write(letter_set=[0,4,5],begin_time=t0+0.7,transition_time=0.05)
        t0+=0.75
        line0.move_copy_to_and_remove(line1, src_letter_indices=[9,10,11,12], target_letter_indices=[7,8,9,10],
                                      begin_time=t0, transition_time=0.7)
        line1.write(letter_set=[ 6,11,12], begin_time=t0 + 0.7, transition_time=0.05)

        t0 += 0.75
        line1.move_copy_to_and_remove(target=line1,
                                      src_letter_indices=[1, 7, 8], target_letter_indices=[13, 14, 15], begin_time=t0,detour=1.5*display2.down)
        t0 += 1.5
        line1.move_copy_to_and_remove(src_letter_indices=[2, 3, 10],
                                      target_letter_indices=[16, 17, 19], begin_time=t0,detour=1.5*display2.down)
        line1.move_copy_to_and_remove(src_letter_indices=[5],
                                      target_letter_indices=[18],
                                      new_color='important', begin_time=t0,detour=1.5*display2.down)
        t0 += 1.5
        line1.write(letter_set=[20, 21], begin_time=t0, transition_time=0.5)
        t0 += 1.5

        # subtraction
        color2 = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3,
             ['joker'] * 2, ['text'], ['important'],
             ['text'] * 2, ['joker'] * 3, ['important'] * 4, ['text'], ['joker'], ['important'] * 3])

        line2 = SimpleTexBObject(r"(3-i)-(-1+i)=3+1-i-i=4-2i", color=color2, name='line2', bevel=0)
        indent = 0.25
        display2.add_text_in(line2, line=2.5, indent=indent, column=1)

        line2.write(letter_range=[0, 13], begin_time=t0)
        t0 += 1.5
        # real part
        line2.move_copy_to_and_remove(src_letter_indices=[1, 7, 8], target_letter_indices=[13, 14, 15], begin_time=t0,detour=1.5*display2.down)
        line2.move_copy_to_and_remove(src_letter_indices=[5], target_letter_indices=[14], begin_time=t0,
                                      new_color='joker',detour=1.5*display2.down)
        t0 += 1.5

        # imaginary part
        line2.move_copy_to_and_remove(src_letter_indices=[2, 3, 10],
                                      target_letter_indices=[16, 17, 19], begin_time=t0,detour=1.5*display2.down)
        line2.move_copy_to_and_remove(src_letter_indices=[5],
                                      target_letter_indices=[18],
                                      new_color='important', begin_time=t0,detour=1.5*display2.down)
        t0 += 1.5
        line2.write(letter_set=[20, 21, 22, 23, 24], begin_time=t0, transition_time=0.5)
        t0 += 1.5

        display3 = Display(scales=[8, 3], number_of_lines=5.5, location=[9, 0, -3], columns=1, flat=True)
        title = SimpleTexBObject(r"\text{Multiplication and Division}", color='text', aligned="center")
        display3.set_title(title)

        display3.appear(begin_time=t0)
        title.write(begin_time=t0, transition_time=2)
        t0 += 2.5

        color1 = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3,
             ['joker'] * 2, ['text'], ['important'],
             ['text'] * 2, ['joker'] * 2, ['text'], ['important'] * 7])
        color2 = flatten(
            [['text'], ['joker'], ['important'] * 2, ['text'] * 3,
             ['joker'] * 2, ['important'] * 2,
             ['text'] * 2, ['joker'] * 2, ['text'], ['important'] * 4, ['joker'] * 2,
             ['text'], ['joker'] * 2, ['text'],
             ['important'] * 2])

        line3 = SimpleTexBObject(r"(3-i)\cdot(-1+i)=-3+3i+i-i^2", color=color1, name='line3', bevel=0)
        line4 = SimpleTexBObject(r"(3-i)\cdot(-1+i)=-3+3i+i+1=-2+4i", color=color2, name='line4', bevel=0)
        indent = 0.25
        display3.add_text_in(line3, line=0.5, indent=indent, column=1, scale=0.65)
        display3.add_text_in(line4, line=0.5, indent=indent, column=1, scale=0.65)

        # multiplication

        line3.write(letter_range=[0, 13], begin_time=t0)
        t0 += 1.5
        line3.move_copy_to_and_remove(src_letter_indices=[1, 7, 8],
                                      target_letter_indices=[14, 13, 14], begin_time=t0,detour=1.5*display3.down)
        t0 += 1.5
        line3.write(letter_set=[15], begin_time=t0, transition_time=0)
        line3.move_copy_to_and_remove(src_letter_indices=[1],
                                      target_letter_indices=[16], begin_time=t0, new_color='important',detour=1.5*display3.down)
        line3.move_copy_to_and_remove(src_letter_indices=[10],
                                      target_letter_indices=[17], begin_time=t0,detour=1.5*display3.down)
        t0 += 1.5
        line3.move_copy_to_and_remove(src_letter_indices=[2, 3],
                                      target_letter_indices=[18, 19], begin_time=t0,detour=1.5*display3.down)
        line3.move_copy_to_and_remove(src_letter_indices=[7, 8],
                                      target_letter_indices=[18, 19],
                                      new_color='important', begin_time=t0,detour=1.5*display3.down)
        t0 += 1
        line3.replace2(line4, src_letter_range=[18, 19], img_letter_range=[18, 19], begin_time=t0, transition_time=0.3,
                       morphing=False)
        t0 += 2
        line3.move_copy_to_and_remove(src_letter_indices=[2, 3, 10],
                                      target_letter_indices=[20, 21, 21], begin_time=t0,detour=1.5*display3.down)
        t0 += 1
        line3.write(letter_set=[22], begin_time=t0, transition_time=0)
        t0 += 0.5
        line3.replace2(line4, src_letter_range=[20, 23], img_letter_range=[20, 22], begin_time=t0, morphing=False)
        t0 += 1.5
        line4.write(letter_set=[22], begin_time=t0, transition_time=0)
        line3.move_copy_to_and_remove(line4, src_letter_indices=[13, 14], target_letter_indices=[23, 24], begin_time=t0,detour=1.5*display3.down)
        line4.move_copy_to_and_remove(line4, src_letter_indices=[20, 21], target_letter_indices=[23, 24], begin_time=t0,detour=1.5*display3.down)
        t0 += 1.5
        line4.write(letter_set=[25], begin_time=t0, transition_time=0)
        line3.move_copy_to_and_remove(line4, src_letter_indices=[16, 17, 19], target_letter_indices=[26, 27, 27],
                                      begin_time=t0,detour=1.5*display3.down)
        t0 += 1.5
        line3.perform_morphing()
        # division

        color5 = flatten(
            [['text'] * 2, ['joker'] * 3, ['important'], ['text'],
             ['important'] * 2, ['text'], ['important'], ['text']])
        color6 = flatten(
            [['text'] * 2, ['joker'] * 3, ['important'], ['text'], ['important'], ['text'], ['important'],
             ['text'] * 5, ['joker'], ['text'],
             ['joker'] * 3, ['important'] * 3,
             ['text'], ['important'], ['text'] * 2, ['joker'] * 4, ['important'] * 5, ['text'], ['important'] * 4,
             ['important'] * 6])
        color7 = flatten(
            [['text'] * 2, ['joker'] * 3, ['important'], ['text'], ['important'], ['text'], ['important'],
             ['text'] * 5, ['joker'], ['text'],
             ['joker'] * 3, ['important'] * 3,
             ['text'], ['important'], ['text'] * 2, ['joker'] * 4, ['important'] * 5, ['text'], ['important'] * 4,
             ['important'] * 7])
        color8 = flatten(
            [['text'] * 2, ['joker'] * 3, ['important'], ['text'], ['important'], ['text'], ['important'],
             ['text'] * 5, ['joker'], ['text'],
             ['joker'] * 3, ['important'] * 3,
             ['text'], ['important'], ['text'] * 2, ['joker'] * 4, ['important'] * 5, ['text'], ['important'] * 4,
             ['joker'] * 5])
        color9 = flatten(
            [['text'] * 2, ['joker'] * 3, ['important'], ['text'], ['important'], ['text'], ['important'],
             ['text'] * 5,
             ['joker'], ['text'],
             ['joker'] * 3, ['important'] * 3,
             ['text'], ['important'], ['text'] * 2, ['joker'] * 4, ['important'] * 5, ['text'], ['important'] * 4,
             ['joker'] * 4, ['text'], ['joker'] * 2, ['text'],['joker'], ['important']])
        color10 = flatten(
            [['text'] * 2, ['joker'] * 3, ['important'], ['text'], ['important'], ['text'], ['important'],
             ['text'] * 5,
             ['joker'], ['text'],
             ['joker'] * 3, ['important'] * 3,
             ['text'], ['important'], ['text'] * 2, ['joker'] * 4, ['important'] * 5, ['text'], ['important'] * 4,
             ['joker'] * 4, ['text'], ['joker'] * 2, ['important'] * 2])

        line5 = SimpleTexBObject(r"{(3-i)\over (-1+i)}", color=color5, name='line5', bevel=0)
        line6 = SimpleTexBObject(r"{(3-i)\cdot(-1-i) \over(-1+i)\cdot (-1-i)}={-3 -3i-i-i^2\over -1 -i -i -i^2}",
                                 color=color6, name='line6', bevel=0)
        line7 = SimpleTexBObject(r"{(3-i)\cdot(-1-i) \over(-1+i)\cdot (-1-i)}={-3 -3i+i+i^2\over +1 +i -i -i^2}",
                                 color=color7, name='line7', bevel=0)
        line8 = SimpleTexBObject(r"{(3-i)\cdot(-1-i) \over(-1+i)\cdot (-1-i)}={-3 -3i+i-1\over +1 +i -i +1}",
                                 color=color8, name='line8', bevel=0)
        line9 = SimpleTexBObject(
            r"{(3-i)\cdot(-1-i) \over(-1+i)\cdot (-1-i)}={-3 -3i+i-1\over +1 +i -i +1}={-4-2i\over 2}",
            color=color9, name='line9', bevel=0)
        line10 = SimpleTexBObject(
            r"{(3-i)\cdot(-1-i) \over(-1+i)\cdot (-1-i)}={-3 -3i+i-1\over +1 +i -i +1}=-2-i",
            color=color10, name='line10', bevel=0)
        indent = 0.25
        display3.add_text_in(line5, line=2.5, indent=indent, column=1, scale=0.65)
        display3.add_text_in(line6, line=2.5, indent=indent, column=1, scale=0.65)
        display3.add_text_in(line7, line=2.5, indent=indent, column=1, scale=0.65)
        display3.add_text_in(line8, line=2.5, indent=indent, column=1, scale=0.65)
        display3.add_text_in(line9, line=2.5, indent=indent, column=1, scale=0.65)
        display3.add_text_in(line10, line=2.5, indent=indent, column=1, scale=0.65)

        line6.write(letter_set=[1, 3, 5, 7, 8], begin_time=t0, transition_time=0.5)
        t0 += 0.75
        line5.write(letter_set=[6], begin_time=t0, transition_time=0.5)
        t0 += 0.75
        line6.write(letter_set=[0, 2, 4, 6, 9, 10], begin_time=t0, transition_time=0.5)
        t0 += 1
        line6.write(letter_set=[12], begin_time=t0, transition_time=0.5)
        line5.letters[6].disappear(begin_time=t0 + 0.5, transition_time=0)
        t0 += 0.75

        #complex conjugation

        display4 = Display(number_of_lines=8, scales=[0.7*4, 0.7*2], columns=1, location=[-2.5, 0,-3], name='DisplayCC', flat=True)
        title4 = SimpleTexBObject(r"\text{Complex Conjugation}", aligned="center", name='Title', color='text')
        display4.set_title(title4,shift=[-0.5,0])

        display4.appear(begin_time=t0, transition_time=0.5)
        title4.write(begin_time=t0, transition_time=0.5)

        colors = flatten([['joker']*2,['text'],['important'], ['text'], ['joker']*2,['important']*2])
        lines = [
            SimpleTexBObject(z2str(z2)+r'\rightarrow'+z2str(np.conj(z2)), color=colors),
        ]

        t0+=0.5
        indents = [1]
        rows = [2]
        scales = [2*0.7]
        for i, line in enumerate(lines):
            display4.add_text_in(line, line=rows[i], indent=indents[i], scale=scales[i])
            line.write(begin_time=t0, transition_time=0.5)

        t0 += 0.6
        line6.write(letter_set=[13, 16, 17, 19, 21, 24, 25], begin_time=t0, transition_time=0.3)
        line6.write(letter_set=[11, 14, 15, 18, 20, 22, 23], begin_time=t0, transition_time=0.3)
        t0 += 0.4

        # first multiplication
        line7.write(letter_set=[26, 36], begin_time=t0, transition_time=0.5)
        t0 += 1
        line6.move_copy_to_and_remove(src_letter_indices=[3, 15, 18],
                                      target_letter_indices=[29, 27, 29], begin_time=t0,detour=1.5*display3.up)
        t0 += 1.5
        line6.move_copy_to_and_remove(src_letter_indices=[3],
                                      target_letter_indices=[33], begin_time=t0, new_color='important',detour=1.5*display3.up)
        line6.move_copy_to_and_remove(src_letter_indices=[20, 22],
                                      target_letter_indices=[31, 35], begin_time=t0,detour=1.5*display3.up)
        t0 += 1.5
        line6.move_copy_to_and_remove(src_letter_indices=[5, 7],
                                      target_letter_indices=[38, 40], begin_time=t0,detour=1.5*display3.up)
        line6.move_copy_to_and_remove(src_letter_indices=[15, 18],
                                      target_letter_indices=[38, 40],
                                      new_color='important', begin_time=t0,detour=1.5*display3.up)
        t0 += 1
        line6.replace2(line7, src_letter_range=[38, 39], img_letter_range=[38, 39], begin_time=t0, transition_time=0.5,
                       morphing=False)
        t0 += 0.5

        line6.move_copy_to_and_remove(src_letter_indices=[5, 7, 20, 22],
                                      target_letter_indices=[42, 44, 42, 44], begin_time=t0,detour=1.5*display3.up)
        t0 += 1
        line6.write(letter_set=[46], begin_time=t0, transition_time=0)
        line6.replace2(line7, src_letter_range=[42, 43], img_letter_range=[42, 43], begin_time=t0, transition_time=0.5,
                       morphing=False)
        t0 += 1.5

        # second multiplication

        line6.move_copy_to_and_remove(src_letter_indices=[2, 4, 17, 19],
                                      target_letter_indices=[28, 30, 28, 30], begin_time=t0,detour=1.5*display3.down)
        t0 += 1
        line6.replace2(line7, src_letter_range=[28, 29], img_letter_range=[28, 29], begin_time=t0, transition_time=0.5,
                       morphing=False)
        t0 += 1
        line6.move_copy_to_and_remove(src_letter_indices=[2, 4],
                                      target_letter_indices=[32, 34], begin_time=t0, new_color='important',detour=1.5*display3.down)
        line6.move_copy_to_and_remove(src_letter_indices=[21, 24],
                                      target_letter_indices=[32, 34], begin_time=t0,detour=1.5*display3.down)
        t0 += 1
        line6.replace2(line7, src_letter_range=[32, 33], img_letter_range=[32, 33], begin_time=t0, transition_time=0.5,
                       morphing=False)
        t0 += 1

        line6.move_copy_to_and_remove(src_letter_indices=[9],
                                      target_letter_indices=[39], begin_time=t0,detour=1.5*display3.down)
        line6.move_copy_to_and_remove(src_letter_indices=[17, 19],
                                      target_letter_indices=[37, 39],
                                      new_color='important', begin_time=t0,detour=1.5*display3.down)
        t0 += 1.5
        line6.move_copy_to_and_remove(src_letter_indices=[9, 21, 24],
                                      target_letter_indices=[43, 41, 43], begin_time=t0,detour=1.5*display3.down)
        t0 += 1
        line6.write(letter_set=[45], begin_time=t0, transition_time=0)
        t0 += 0.5

        line6.perform_morphing()
        # convert i^2->-1
        set = [41, 43, 44, 45, 46]
        t0 += 1.5
        line7.write(letter_set=set, begin_time=t0, transition_time=0)
        for s in set:
            line6.letters[s].disappear(begin_time=t0 + 0.1, transition_time=0)

        line7.replace2(line8, src_letter_range=[41, 47], img_letter_range=[41, 45], begin_time=t0)
        t0 += 1.5

        line9.write(letter_set=[45, 48], begin_time=t0, transition_time=0.5)
        t0 += 1
        line6.move_copy_to_and_remove(line9, src_letter_indices=[27, 29], target_letter_indices=[46, 47], begin_time=t0,detour=1.5*display3.up)
        line8.move_copy_to_and_remove(line9, src_letter_indices=[42, 44], target_letter_indices=[46, 47], begin_time=t0,detour=1.5*display3.up)
        t0 += 1.5
        line6.move_copy_to_and_remove(line9, src_letter_indices=[31, 33, 35, 40],
                                      target_letter_indices=[50, 51, 52, 52], begin_time=t0,detour=1.5*display3.up)
        t0 += 1.2
        set = [34,37,39]
        for s in set:
            line6.letters[s].change_color(new_color='example',begin_time=t0,transition_time=0.5)
            line6.letters[s].disappear(begin_time=t0+0.6,transitions_time=0.5)
        line7.letters[32].change_color(new_color='example', begin_time=t0, transition_time=0.5)
        line7.letters[32].disappear(begin_time=t0 + 0.6, transitions_time=0.5)

        line6.move_copy_to_and_remove(line9, src_letter_indices=[30], target_letter_indices=[49], begin_time=t0+1.2,transition_time=0.5,detour=1.5*display3.down)
        line8.move_copy_to_and_remove(line9, src_letter_indices=[43], target_letter_indices=[49], begin_time=t0+1.2,transition_time=0.5,detour=1.5*display3.down)
        t0 += 1.80

        line9.replace2(line10, src_letter_range=[46, 53], img_letter_range=[46, 50], begin_time=t0)

        t0 += 1.5

        title2b = SimpleTexBObject(r"\text{Addition and Subtraction}", color='text', aligned="center")
        display2.set_title_back(title2b)
        title2b.write(begin_time=t0, transition_time=0)
        display2.turn(flat=True, begin_time=t0,transition_time=0.5)
        display4.disappear(begin_time=t0,transition_time=0.5)

        # make multiplication disappear
        line3.disappear(begin_time=t0,transition_time=0.5)
        line4.disappear(begin_time=t0,transition_time=0.5)
        t0+=1

        title3b = SimpleTexBObject(r"\text{Multiplication and Division}", color='text', aligned="center")
        display3.set_title_back(title3b)
        title3b.write(begin_time=t0, transition_time=0)


        display3.turn(flat=True, begin_time=t0, transition_time=0.5,reverse=True)
        t0 += 1

        print('last frame ', t0 * FRAME_RATE)

    def notation(self):
        cues = self.sub_scenes['notation']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=20, columns=1, location=[12.5, 0, 0], name='Display')
        title = SimpleTexBObject(r"\text{Some Notation}", aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)

        details = 3
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 4], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 4.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[-2, 0, 0],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)
        t0 += 2.5

        lines = [
            SimpleTexBObject(r"\text{The absolute value }|a+bi|"),
            SimpleTexBObject(r"a+bi"),
            SimpleTexBObject(r"|a+bi|=\sqrt{a^2+b^2}", color='example')
        ]

        for i, line in enumerate(lines):
            display.add_text_in(line, line=i + 1, indent=1)

        lines[0].write(begin_time=t0)
        t0 += 1.5

        sphere = Sphere(0.25, location=z2loc(coords, 4 + 3j), name='4+3i')
        coords.add_object(sphere)
        sphere.grow(begin_time=t0)
        sphere.write_name_as_label(begin_time=t0 + 0.5, transition_time=0.5, modus='right', offset=[0.4, 0, 0],
                                   aligned='left')
        #lines[1].write(begin_time=t0)
        t0 += 1.5

        a = Cylinder.from_start_to_end(
            end=z2loc(coords, 0),
            start=z2loc(coords, 4),
            color='joker',
            name='a=4',
            label_rotation=[np.pi/2,np.pi/2,0]
        )
        coords.add_object(a)
        a.grow(modus='from_end', begin_time=t0)
        a.write_name_as_label(modus='right', begin_time=t0 + 0.5, offset=[3, 0, -a.length / 6])

        t0 += 1

        b = Cylinder.from_start_to_end(
            end=z2loc(coords, 4),
            start=z2loc(coords, 4 + 3j),
            color='important',
            name='b=3',
            label_rotation=[np.pi / 2, -np.pi / 2, 0],
        )
        coords.add_object(b)
        b.grow(modus='from_end', begin_time=t0)
        b.write_name_as_label(modus='right', begin_time=t0 + 0.5, offset=[-2.5, 0, -b.length / 6])

        t0 += 1

        c = Cylinder.from_start_to_end(
            end=z2loc(coords, 0),
            start=z2loc(coords, 4 + 3j),
            color='example',
            name=r'c=\sqrt{4^2+3^2}=5',
        )
        coords.add_object(c)
        c.grow(modus='from_end', begin_time=t0)
        c.write_name_as_label(modus='right', begin_time=t0 + 0.5, offset=[-4, -0, -c.length / 8])
        lines[2].write(begin_time=t0)

        t0 += 1.5

        poly1 = Polygon(vertices=[z2loc(coords, 0), z2loc(coords, 4), z2loc(coords, 4 - 4j), z2loc(coords, -4j)],
                        edges=[[0, 1], [1, 2], [2, 3], [3, 0]], name=r'a^2', color='joker')
        coords.add_object(poly1)
        poly1.appear(begin_time=t0, transition_time=2)

        poly2 = Polygon(vertices=[z2loc(coords, 4), z2loc(coords, 4 + 3j), z2loc(coords, 7 + 3j), z2loc(coords, 7)],
                        edges=[[0, 1], [1, 2], [2, 3], [3, 0]], name=r'b^2', color='important')
        coords.add_object(poly2)
        poly2.appear(begin_time=t0, transition_time=2)

        poly3 = Polygon(
            vertices=[z2loc(coords, 0), z2loc(coords, 4 + 3j), z2loc(coords, 1 + 7j), z2loc(coords, -3 + 4j)],
            edges=[[0, 1], [1, 2], [2, 3], [3, 0]], name=r'b^2', color='example')
        coords.add_object(poly3)
        poly3.appear(begin_time=t0, transition_time=2)

        coords.move(direction=[-5, 15, -2], begin_time=t0)
        t0 += 2.5

        coords.move(direction=[5, -15, 2], begin_time=t0)
        poly1.disappear(begin_time=t0, transition_time=0.5)
        poly2.disappear(begin_time=t0, transition_time=0.5)
        poly3.disappear(begin_time=t0, transition_time=0.5)
        t0 += 1.5

        lines2 = [
            SimpleTexBObject(r"\text{The argument }arg(a+bi)",color=flatten([['text']*11,['example']]))
        ]

        for i, line in enumerate(lines2):
            display.add_text_in(line, line=2 * i + 6, indent=1, scale=0.7 - 0.2 * i)

        lines2[0].write(begin_time=t0)
        t0 += 1.5

        arg_z = CircleArc(center=z2loc(coords, 0), radius=2.5, start_angle=0, end_angle=np.arctan(3 / 4),
                          color='example', name=r'\arg(a+bi)',
                          mode='XZ', thickness=0.5)
        coords.add_object(arg_z)
        arg_z.grow(begin_time=t0)
        arg_z.write_name_as_label(modus='center', begin_time=t0 + 0.5, transition_time=0.5)
        t0 += 1.5

        print(t0)

    def wlog4(self):
        cues = self.sub_scenes['wlog4']
        t0 = 0  # cues['start']

        ibpy.set_sun_light(location=[0, -10, 35])
        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        plane = Plane(u=[-14, 14], v=[-10, 10], color='sand',
                      location=[0, 0, -10])  # Watch out, the location is vertex location that gets rescaled
        plane.appear(begin_time=t0)
        ibpy.add_modifier(plane, type='COLLISION')
        info = InfoPanel(location=[-7, -5, -10], rotation_euler=[0, 0, np.pi + np.pi / 4],
                         colors=['marble', 'background'])
        info.ref_obj.parent = plane.ref_obj
        info.appear(begin_time=t0)
        empty.move(direction=[0, 0, -10], begin_time=t0, transition_time=5)
        t0 += 5.5

        t0 += 1.5

        philosophers = [
            PersonWithCape(location=[-4, -6, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_8', 'drawing'],
                           simulation_start=0, simulation_duration=40, name='Greek1'),
            PersonWithCape(location=[-6, -8, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_7', 'important'],
                           simulation_start=0, simulation_duration=40, name='Greek2'),
        ]

        for i in range(0, 2):
            philosophers[i].appear(begin_time=t0)

        t0 += 2
        # background philosopher
        philosophers[0].rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0.5)
        philosophers[0].move(direction=[2, 13, 0], begin_time=t0 + 0.5)
        philosophers[0].rotate(rotation_euler=[0, 0, -3 * np.pi / 4], begin_time=t0 + 1.5, transition_time=0.5)

        # forground philosopher
        philosophers[1].rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=t0, transition_time=0.5)
        philosophers[1].move(direction=[11, 0, 0], begin_time=t0 + 0.5)
        philosophers[1].rotate(rotation_euler=[0, 0, np.pi / 4], begin_time=t0 + 1.5, transition_time=0.5)

        t0 += 2.5
        # background
        l0 = Vector([-1, 6, -10])
        # forground
        l1 = Vector([4, -7, -10])

        pencil = Pencil(location=l0, colors=['wood', 'drawing'], name='Pencil1')
        pencil2 = Pencil(location=l1, colors=['wood', 'important'], name='Pencil2')

        pencil.appear(begin_time=t0)
        t0 += 1.5
        pencil.rotate(rotation_euler=[np.pi / 4, 0, 0], begin_time=t0)

        duration = 5
        geometry1 = action2(pencil, l0,
                            Vector([0, -1, 0]),
                            Vector([1, 0, 0]), 4,
                            'drawing', t0, duration,
                            "Geo1", case=0)
        geometry1.appear(begin_time=t0)
        t0 += duration + 1
        pencil.disappear(begin_time=t0)

        t0 -= duration + 1
        pencil2.appear(begin_time=t0)
        pencil2.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0)

        geometry2 = action2(pencil2, l1,
                            Vector([-1, 0, 0]),
                            Vector([0, 1, 0]), 8,
                            'important', t0, duration,
                            "Geo2", case=1)
        geometry2.appear(begin_time=t0)
        t0 += duration + 1

        pencil2.disappear(begin_time=t0)
        philosophers[0].disappear(begin_time=t0, transition_time=0.5)
        philosophers[1].disappear(begin_time=t0, transition_time=0.5)
        philosophers[0].hide(begin_time=t0 + 0.3)
        philosophers[1].hide(begin_time=t0 + 0.3)
        info.disappear(begin_time=t0)
        t0 += 1
        #
        geometry1.ref_obj.parent = plane.ref_obj
        geometry2.ref_obj.parent = plane.ref_obj

        plane.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=2)
        empty.move_to(target_location=[0, 0, 0], begin_time=t0, transition_time=2)
        t0 += 2.5
        print(t0 * FRAME_RATE)

        # scale
        geometry1.rescale(rescale=[2, 2, 1], begin_time=t0, transition_time=1)
        t0 += 1.5

        # rotate
        geometry1.rotate(rotation_euler=[0, 0, np.pi / 2], begin_time=t0)
        t0 += 1.5

        # move
        geometry1.move(direction=[0, -7, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # move
        geometry2.move(direction=[-5, 6, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # mirror
        geometry2.move(direction=[0, 0, 0.025], begin_time=t0)
        geometry2.rotate(rotation_euler=[0, np.pi, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        geometry1.rotate(rotation_euler=[0, np.pi, np.pi / 2], begin_time=t0)
        t0 += 1.5

        print(t0)

    def wlog4b(self):
        cues = self.sub_scenes['wlog4b']
        t0 = 0  # cues['start']

        ibpy.set_sun_light(location=[0, -10, 35])
        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        plane = Plane(u=[-14, 14], v=[-10, 10], color='sand',
                      location=[0, 0, -10])  # Watch out, the location is vertex location that gets rescaled
        plane.appear(begin_time=t0)
        ibpy.add_modifier(plane, type='COLLISION')
        info = InfoPanel(location=[-7, -5, -10], rotation_euler=[0, 0, np.pi + np.pi / 4],
                         colors=['marble', 'background'])
        info.ref_obj.parent = plane.ref_obj
        info.appear(begin_time=t0)
        empty.move(direction=[0, 0, -10], begin_time=t0, transition_time=5)
        t0 += 5.5

        t0 += 1.5

        philosophers = [
            PersonWithCape(location=[-4, -6, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_8', 'drawing'],
                           simulation_start=5, simulation_duration=25, name='Greek1'),
            PersonWithCape(location=[-6, -8, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_7', 'important'],
                           simulation_start=5, simulation_duration=25, name='Greek2'),
        ]

        for i in range(0, 2):
            philosophers[i].appear(begin_time=t0)

        t0 += 2
        # background philosopher
        philosophers[0].rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0.5)
        philosophers[0].move_to(target_location=[-4, -2, -9.95], begin_time=t0 + 0.5)

        # forground philosopher
        philosophers[1].rotate(rotation_euler=[0, 0, -np.pi / 180 * 48.37], begin_time=t0, transition_time=0.5)
        philosophers[1].move_to(target_location=[5, -2, -9.95], begin_time=t0 + 0.5)
        philosophers[1].rotate(rotation_euler=[0, 0, 0], begin_time=t0 + 1.5, transition_time=0.5)

        t0 += 2.5
        # background
        l0 = Vector([-5, -1, -10])
        # forground
        l1 = Vector([4, -1, -10])

        pencil = Pencil(location=l0, colors=['wood', 'drawing'], name='Pencil1')
        pencil2 = Pencil(location=l1, colors=['wood', 'important'], name='Pencil2')

        pencil.appear(begin_time=t0)
        pencil2.appear(begin_time=t0)
        t0 += 1.5

        pencil.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0)
        pencil2.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0)

        duration = 5
        geometry1 = action3(pencil, l0,
                            Vector([1, 0, 0]),
                            Vector([0, -1, 0]), 8,
                            'drawing', t0, duration,
                            "Geo1", case=0)
        geometry1.appear(begin_time=t0)

        geometry2 = action3(pencil2, l1,
                            Vector([1, 0, 0]),
                            Vector([0, 1, 0]), 8,
                            'important', t0, duration,
                            "Geo2", case=0)

        geometry2.appear(begin_time=t0)

        t0 += duration + 1
        pencil.disappear(begin_time=t0)
        pencil2.disappear(begin_time=t0)

        philosophers[0].disappear(begin_time=t0, transition_time=0.5)
        philosophers[1].disappear(begin_time=t0, transition_time=0.5)
        philosophers[0].hide(begin_time=t0 + 0.3)
        philosophers[1].hide(begin_time=t0 + 0.3)
        info.disappear(begin_time=t0)
        t0 += 1
        #
        geometry1.ref_obj.parent = plane.ref_obj
        geometry2.ref_obj.parent = plane.ref_obj

        plane.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=2)
        empty.move_to(target_location=[0, 0, 0], begin_time=t0, transition_time=2)
        t0 += 2.5

        # move
        geometry2.move(direction=[-9, 0, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # mirror
        geometry2.move(direction=[0, 0, 0.025], begin_time=t0)
        geometry2.rotate(rotation_euler=[-np.pi, 0, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        print(t0 * FRAME_RATE)

    def wlog3(self):
        cues = self.sub_scenes['wlog3']
        t0 = 0  # cues['start']

        ibpy.set_sun_light(location=[0, -10, 35])
        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        plane = Plane(u=[-14, 14], v=[-10, 10], color='sand',
                      location=[0, 0, -10])  # Watch out, the location is vertex location that gets rescaled
        plane.appear(begin_time=t0)
        ibpy.add_modifier(plane, type='COLLISION')
        info = InfoPanel(location=[-7, -5, -10], rotation_euler=[0, 0, np.pi + np.pi / 4],
                         colors=['marble', 'background'])
        info.ref_obj.parent = plane.ref_obj
        info.appear(begin_time=t0)
        empty.move(direction=[0, 0, -10], begin_time=t0, transition_time=5)
        t0 += 5.5

        t0 += 1.5

        philosophers = [
            PersonWithCape(location=[-4, -6, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_8', 'drawing'],
                           simulation_start=0, simulation_duration=40, name='Greek1'),
            PersonWithCape(location=[-6, -8, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_7', 'important'],
                           simulation_start=0, simulation_duration=40, name='Greek2'),
        ]

        for i in range(0, 2):
            philosophers[i].appear(begin_time=t0)

        t0 += 2
        # background philosopher
        philosophers[0].rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0.5)
        philosophers[0].move(direction=[2, 13, 0], begin_time=t0 + 0.5)
        philosophers[0].rotate(rotation_euler=[0, 0, -3 * np.pi / 4], begin_time=t0 + 1.5, transition_time=0.5)

        # forground philosopher
        philosophers[1].rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=t0, transition_time=0.5)
        philosophers[1].move(direction=[11, 0, 0], begin_time=t0 + 0.5)
        philosophers[1].rotate(rotation_euler=[0, 0, np.pi / 4], begin_time=t0 + 1.5, transition_time=0.5)

        t0 += 2.5
        # background
        l0 = Vector([-1, 6, -10])
        # forground
        l1 = Vector([4, -7, -10])

        pencil = Pencil(location=l0, colors=['wood', 'drawing'], name='Pencil1')
        pencil2 = Pencil(location=l1, colors=['wood', 'important'], name='Pencil2')

        pencil.appear(begin_time=t0)
        t0 += 1.5
        pencil.rotate(rotation_euler=[np.pi / 4, 0, 0], begin_time=t0)

        duration = 5
        geometry1 = action(pencil, l0,
                           Vector([0, -1, 0]),
                           Vector([1, 0, 0]), 5,
                           'drawing', t0, duration,
                           "Geo1", case=0)
        geometry1.appear(begin_time=t0)
        t0 += duration + 1
        pencil.disappear(begin_time=t0)

        t0 -= duration + 1
        pencil2.appear(begin_time=t0)
        pencil2.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0)

        geometry2 = action(pencil2, l1,
                           Vector([-1, 0, 0]),
                           Vector([0, 1, 0]), 10,
                           'important', t0, duration,
                           "Geo2", case=1)
        geometry2.appear(begin_time=t0)
        t0 += duration + 1

        pencil2.disappear(begin_time=t0)
        philosophers[0].disappear(begin_time=t0, transition_time=0.5)
        philosophers[1].disappear(begin_time=t0, transition_time=0.5)
        philosophers[0].hide(begin_time=t0 + 0.3)
        philosophers[1].hide(begin_time=t0 + 0.3)
        info.disappear(begin_time=t0)
        t0 += 1
        #
        geometry1.ref_obj.parent = plane.ref_obj
        geometry2.ref_obj.parent = plane.ref_obj

        plane.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=2)
        empty.move_to(target_location=[0, 0, 0], begin_time=t0, transition_time=2)
        t0 += 2.5
        print(t0 * FRAME_RATE)

        # scale
        geometry1.rescale(rescale=[2, 2, 1], begin_time=t0, transition_time=1)
        t0 += 1.5

        # rotate
        geometry1.rotate(rotation_euler=[0, 0, np.pi / 2], begin_time=t0)
        t0 += 1.5

        # move
        geometry1.move(direction=[0, -7, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # move
        geometry2.move(direction=[-5, 6, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # mirror
        geometry2.rotate(rotation_euler=[0, np.pi, 0], begin_time=t0, transition_time=1)
        geometry2.move(direction=[0, 0, 0.025], begin_time=t0)
        t0 += 1.5

        print(t0)

    def wlog2(self):
        cues = self.sub_scenes['wlog2']
        t0 = 0  # cues['start']

        ibpy.set_sun_light(location=[0, -10, 35])
        ibpy.set_camera_location(location=[0, -20, 0])
        empty = EmptyCube(location=[0, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        plane = Plane(u=[-14, 14], v=[-10, 10], color='sand',
                      location=[0, 0, -10])  # Watch out, the location is vertex location that gets rescaled
        plane.appear(begin_time=t0)
        ibpy.add_modifier(plane, type='COLLISION')
        info = InfoPanel(location=[-7, -5, -10], rotation_euler=[0, 0, np.pi + np.pi / 4],
                         colors=['marble', 'background'])
        info.ref_obj.parent = plane.ref_obj
        info.appear(begin_time=t0)
        empty.move(direction=[0, 0, -10], begin_time=t0, transition_time=5)
        t0 += 5.5

        t0 += 1.5

        philosophers = [
            PersonWithCape(location=[-4, -6, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_8', 'drawing'],
                           simulation_start=0, simulation_duration=40, name='Greek1'),
            PersonWithCape(location=[-6, -8, -9.95], rotation_euler=[0, 0, np.pi / 4],
                           colors=['gray_7', 'important'],
                           simulation_start=0, simulation_duration=40, name='Greek2'),
        ]

        for i in range(0, 2):
            philosophers[i].appear(begin_time=t0)

        t0 += 2
        # background philosopher
        philosophers[0].rotate(rotation_euler=[0, 0, 0], begin_time=t0, transition_time=0.5)
        philosophers[0].move(direction=[2, 13, 0], begin_time=t0 + 0.5)
        philosophers[0].rotate(rotation_euler=[0, 0, -3 * np.pi / 4], begin_time=t0 + 1.5, transition_time=0.5)

        # forground philosopher
        philosophers[1].rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=t0, transition_time=0.5)
        philosophers[1].move(direction=[11, 0, 0], begin_time=t0 + 0.5)
        philosophers[1].rotate(rotation_euler=[0, 0, np.pi / 4], begin_time=t0 + 1.5, transition_time=0.5)

        t0 += 2.5
        # background
        l0 = Vector([-1, 6, -10])
        # forground
        l1 = Vector([4, -7, -10])

        pencil = Pencil(location=l0, colors=['wood', 'drawing'], name='Pencil1')
        pencil2 = Pencil(location=l1, colors=['wood', 'important'], name='Pencil2')

        pencil.appear(begin_time=t0)
        t0 += 1.5
        pencil.rotate(rotation_euler=[np.pi / 4, 0, 0], begin_time=t0)

        duration = 15
        geometry1 = golden_action(pencil, l0, Vector([0, -1, 0]), Vector([1, 0, 0]), 6, 2, 'drawing', t0, duration,
                                  "Geo1")
        geometry1.appear(begin_time=t0)
        t0 += duration + 1
        pencil.disappear(begin_time=t0)

        t0 -= duration + 1
        pencil2.appear(begin_time=t0)
        pencil2.rotate(rotation_euler=[0, -np.pi / 4, 0], begin_time=t0)

        geometry2 = golden_action(pencil2, l1, Vector([-1, 0, 0]), Vector([0, 1, 0]), 3, 2, 'important', t0, duration,
                                  "Geo2", sign=-1)
        geometry2.appear(begin_time=t0)
        t0 += duration + 1

        pencil2.disappear(begin_time=t0)
        philosophers[0].disappear(begin_time=t0, transition_time=0.5)
        philosophers[1].disappear(begin_time=t0, transition_time=0.5)
        philosophers[0].hide(begin_time=t0 + 0.3)
        philosophers[1].hide(begin_time=t0 + 0.3)
        info.disappear(begin_time=t0)
        t0 += 1
        #
        geometry1.ref_obj.parent = plane.ref_obj
        geometry2.ref_obj.parent = plane.ref_obj

        plane.rotate(rotation_euler=[np.pi / 2, 0, 0], begin_time=t0, transition_time=2)
        empty.move_to(target_location=[0, 0, 0], begin_time=t0, transition_time=2)
        t0 += 2.5
        print(t0 * FRAME_RATE)

        # move
        geometry2.move(direction=[-5, 0, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # scale
        geometry2.rescale(rescale=[2, 2, 1], begin_time=t0, transition_time=1)
        t0 += 1.5

        # rotate
        geometry2.rotate(rotation_euler=[0, 0, -np.pi / 2], begin_time=t0)
        t0 += 1.5

        # move
        geometry2.move(direction=[0, 7, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        # mirror
        geometry2.rotate(rotation_euler=[0, np.pi, -np.pi / 2], begin_time=t0, transition_time=1)
        t0 += 1.5

        # move
        geometry2.move(direction=[0, 6, 0], begin_time=t0, transition_time=1)
        t0 += 1.5

        geometry1.move(direction=[0, 0, 0.05], begin_time=t0)
        t0 += 1.5

        print(t0)

    def wlog(self):
        cues = self.sub_scenes['wlog']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # the problem starts

        display0 = Display(scales=[4, 2.5], location=[12.5, 0, 2.5], number_of_lines=9)
        title = SimpleTexBObject(r"\text{Problem 3}", color='important', aligned='center')
        display0.set_title(title)
        title.write(begin_time=t0, transition_time=2)
        t0 += 2

        colors = [
            flatten([['text'] * 3, ['drawing'] * 3, ['text']]),
            flatten([['text'] * 6, ['example'], ['text']]),
            flatten([['text'] * 8, ['drawing'], ['text'] * 6, ['drawing']]),
            flatten([['text'] * 3, ['joker'], ['text'] * 13, ['example']]),
            flatten([['text'] * 6, ['example'] * 13, ['text'] * 2, ['drawing'] * 3]),
            flatten([['text'] * 14, ['example']]),
        ]

        lines = [
            SimpleTexBObject(r"\text{Let $\overline{AB}$ be a line segment}", aligned='left', color=colors[0]),
            SimpleTexBObject(r"\text{and let $T$ be a point on it}", aligned='left', color=colors[1]),
            SimpleTexBObject(r"\text{closer to $B$ than to $A$.}", aligned='left', color=colors[2]),
            SimpleTexBObject(r"\text{Let $C$ be a point on the line}", aligned='left', color=colors[3]),
            SimpleTexBObject(r"\text{that is perpendicular to $\overline{AB}$}", aligned='left', color=colors[4]),
            SimpleTexBObject(r"\text{and goes through $T$.}", aligned='left', color=colors[5]),
        ]

        display0.add_text_in(lines[0], line=1, indent=0.5)
        lines[0].write(begin_time=t0)

        display = Display(number_of_lines=10, columns=1, location=[12.5, 0, -2.5], scales=[4, 2.5], name='Display')
        title = SimpleTexBObject(r"\text{Without loss of generality}", aligned="center", name='Title', color='text')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        # details = 3
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 1], [-1, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[3, 3],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2., 1.1, 1), np.arange(-1, 2.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[2, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)
        coords.draw_grid_lines(colors=['drawing', 'drawing'], begin_time=t0, transition_time=2, sub_grid=5)
        t0 += 2.5

        # move A
        noise = PerlinNoise(octaves=5)
        duration = 10
        frames = duration * FRAME_RATE
        steps = 10
        s_a = Sphere(0.25, location=[-1 + 6 * noise([0, 0.68]), 0, 6 * noise([0.68, 0])], name='A=0')
        s_a.grow(begin_time=t0)
        s_a.write_name_as_label(letter_set=[0], modus='down_left', begin_time=t0 + 0.5)
        coords.add_object(s_a)
        for i in range(0, frames + 1, steps):
            location = [-1 + 6 * noise([i / frames, 0.68]), 0, 6 * noise([0.68, i / frames])]
            s_a.move_to(target_location=location, begin_time=t0 + i / FRAME_RATE, transition_time=steps / FRAME_RATE)

        t0 += duration

        line1 = SimpleTexBObject(r"A=a+b\,i", aligned='left')
        display.add_text_in(line1, line=2, indent=1)
        line1.write(begin_time=t0 - duration / 2)

        line2 = SimpleTexBObject(r"B=c+d\,i", aligned='left')
        display.add_text_in(line2, indent=1, line=3)
        line2.write(begin_time=t0 - duration / 2 + 1.1)

        # move B
        noise = PerlinNoise(octaves=3)
        frames = duration * FRAME_RATE
        steps = 10
        location2 = [1 + 6 * noise([0, 0.19]), 0, 2 + 6 * noise([0.19, 0])]
        s_b = Sphere(0.25, location=location2, name='B=1')
        s_b.grow(begin_time=t0)
        s_b.write_name_as_label(letter_set=[0], begin_time=t0 + 0.5, modus='down_right')
        l_ab = Cylinder.from_start_to_end(start=location, end=location2, thickness=0.5)
        l_ab.grow(begin_time=t0, modus='from_start')
        coords.add_object(s_b)
        coords.add_object(l_ab)
        t0 += 1
        for i in range(0, frames + 1, steps):
            location2 = [1 + 6 * noise([i / frames, 0.19]), 0, 2 + 6 * noise([0.19, i / frames])]
            s_b.move_to(target_location=location2, begin_time=t0 + i / FRAME_RATE, transition_time=steps / FRAME_RATE)
            l_ab.move_end_point(target_location=location2, begin_time=t0 + i / FRAME_RATE,
                                transition_time=steps / FRAME_RATE)

        t0 += duration + 1

        cancel = Cylinder(length=0.5, location=[-0.575, 0.1, 0], thickness=0.1, rotation_euler=[0, np.pi / 2, 0],
                          color='important', name="Cancel1")
        cancel2 = Cylinder(length=0.5, location=[-0.575, 0.3, 0], thickness=0.1, rotation_euler=[0, np.pi / 2, 0],
                           color='important', name="Cancel2")
        display.add_child(cancel)
        display.add_child(cancel2)
        cancel.grow(begin_time=t0, modus='from_center')
        cancel2.grow(begin_time=t0, modus='from_center')

        t0 += 1
        line3 = SimpleTexBObject(r"A=0", aligned='left')
        display.add_text_in(line3, indent=1, line=5)
        line3.write(begin_time=t0)

        l_ab2 = Cylinder.from_start_to_end(start=location2, end=location, thickness=0.5, name='lab2')
        coords.add_object(l_ab2)
        l_ab2.grow(begin_time=t0 - 0.1, transition_time=0.1, modus='from_start')
        l_ab.disappear(begin_time=t0, transition_time=0)

        displacement = Vector() - to_vector(location)
        dt = 0.1
        for i in range(1, 11):
            pos = to_vector(location) + i * 0.1 * displacement
            s_a.move_to(target_location=pos, begin_time=t0 + (i - 1) * dt, transition_time=dt)
            l_ab2.move_end_point(target_location=pos, begin_time=t0 + (i - 1) * dt, transition_time=dt)

        t0 += 1.1
        line4 = SimpleTexBObject(r"B=1", aligned='left')
        display.add_text_in(line4, indent=1, line=6)
        line4.write(begin_time=t0)

        l_ab3 = Cylinder.from_start_to_end(start=Vector(), end=location2, thickness=0.5, name='lab2')
        coords.add_object(l_ab3)
        l_ab3.grow(begin_time=t0 - 0.1, transition_time=0.1, modus='from_start')
        l_ab2.disappear(begin_time=t0, transition_time=0)

        b_loc = coords.coords2location([1, 0])
        displacement = b_loc - to_vector(location2)
        for i in range(1, 11):
            pos = to_vector(location2) + i * 0.1 * displacement
            s_b.move_to(target_location=pos, begin_time=t0 + (i - 1) * dt, transition_time=dt)
            l_ab3.move_end_point(target_location=pos, begin_time=t0 + (i - 1) * dt, transition_time=dt)

        t0 += 1

        title.write(begin_time=t0, transition_time=1)

        title2 = SimpleTexBObject(r"\text{Without loss of generality}", aligned="center", name='Title', color='text')
        display.set_title_back(title2)
        title2.write(begin_time=t0)

        line5 = SimpleTexBObject(r"A=0", aligned='left')
        display.add_text_in_back(line5, indent=1, line=1)
        line5.write(begin_time=t0)

        line6 = SimpleTexBObject(r"B=1", aligned='left')
        display.add_text_in_back(line6, indent=1, line=2)
        line6.write(begin_time=t0)
        t0 += 1.5

        display.rotate(rotation_euler=[np.pi / 2, 0, np.pi - np.pi / 15], begin_time=t0)
        t0 += 1.5

        t_loc = coords.coords2location([0.75, 0])
        s_t = Sphere(0.25, location=t_loc, name='T=x', color='example')
        s_t.write_name_as_label(letter_range=[0,1],begin_time=t0,modus='down_left', offset=[-0.15454, 0, 0])
        s_t.grow(begin_time=t0)
        coords.add_object(s_t)

        for i in range(1, 3):
            display0.add_text_in(lines[i], line=i + 1, indent=0.5)
            lines[i].write(begin_time=t0)
            t0 += 1.1

        line7 = SimpleTexBObject(r"T=x", aligned='left')
        line7b = SimpleTexBObject(r"x\in (0.5,1]", aligned='left')
        display.add_text_in_back(line7, indent=1, line=4)
        display.add_text_in_back(line7b, indent=4, line=4)
        line7.write(begin_time=t0, transition_time=0.5)
        line7b.write(begin_time=t0 + 0.9, transition_time=0.5)

        s_t.write_name_as_label(letter_range=[1,3],begin_time=t0, modus='down_left', offset=[-0.15454, 0, 0])

        t0 += 1.5

        c_loc = coords.coords2location([0.75, 1.5])
        offset = coords.coords2location([0, 0.2])
        s_c = Sphere(0.25, location=c_loc, name='C=x+y\,i', color='joker')
        coords.add_object(s_c)
        s_c.grow(begin_time=t0)
        s_c.write_name_as_label(letter_range=[0,1],begin_time=t0, modus='down_left', offset=[-0.59, 0, 0])

        l_tc = Cylinder.from_start_to_end(start=t_loc - offset, end=c_loc + offset, color='example', thickness=0.5)
        coords.add_object(l_tc)
        l_tc.grow(begin_time=t0 + 1)
        ra = RightAngle(location=t_loc, radius=0.65, thickness=0.4, mode='XZ', color='example',
                        name='RA_C' + str(self.construction_counter))
        ra.appear(begin_time=t0+1.5, transition_time=0.5)
        coords.add_object(ra)

        for i in range(3, 6):
            display0.add_text_in(lines[i], line=i + 1, indent=0.5)
            lines[i].write(begin_time=t0)
            t0 += 1.1

        line8 = SimpleTexBObject(r"C=x+y\,i", aligned='left')
        line8b = SimpleTexBObject(r"y>0", aligned='left')
        display.add_text_in_back(line8, indent=1, line=5)
        display.add_text_in_back(line8b, indent=4, line=5)
        line8.write(begin_time=t0, transition_time=0.5)
        line8b.write(begin_time=t0 + 0.9, transition_time=0.5)
        s_c.write_name_as_label(letter_range=[1,6],begin_time=t0, modus='down_left', offset=[-0.59, 0, 0])

        print(t0)

    def multiplication(self):
        cues = self.sub_scenes['multiplication']
        t0 = 0.5  # cues['start']

        z1 = 3 - 1j
        z2 = -1 + 1j

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=15, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{Multiplication}", color='text', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 4], [-2, 4]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2, 4.1, 1), np.arange(-2, 4.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[-2, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)

        t0 += 3

        lines = [
            SimpleTexBObject(r'z_1=' + z2str(z1), color='drawing'),
            SimpleTexBObject(r'|z_1|=\sqrt{10}\approx3.3', color='drawing'),
            SimpleTexBObject(r'\arg(z_1)=-\arctan\left({1\over3}\right)\approx-18^\circ', color='drawing'),
            SimpleTexBObject(r'z_2=' + z2str(z2), color='example'),
            SimpleTexBObject(r'|z_1|=\sqrt{2}\approx1.4', color='example'),
            SimpleTexBObject(r'\arg(z_2)=-\arctan\left(1\right)+180^\circ=135^\circ', color='example'),
            SimpleTexBObject(r'z_1\cdot z_2=' + z2str(z1 * z2), color='important'),
            SimpleTexBObject(r'|z_1\cdot z_2|=\sqrt{10}\cdot \sqrt{2}\approx4.5', color='important'),
            SimpleTexBObject(r'\arg(z_1\cdot z_2)\approx -18^\circ+135^\circ=117^\circ', color='important'),
        ]
        spheres = [
            Sphere(0.25, location=coords.coords2location(z2p(z1)), smooth=2, name='Sz1'),
            Sphere(0.25, location=coords.coords2location(z2p(z2)), smooth=2, name='Sz2', color='example'),
            Sphere(0.25, location=coords.coords2location(z2p(z1 * z2)), smooth=2, name='Sz1z2', color='important'),
            Sphere(0.26, location=coords.coords2location(z2p(z1)), smooth=2, name='Mover', color='important'),

        ]
        coords.add_objects(spheres)

        for i, line in enumerate(lines):
            if i % 3 == 0:
                scale = 0.7
                indent = 1
            else:
                scale = 0.5
                indent = 1.5
            display.add_text_in(line, line=i + 1 + int(i / 3) * 1, indent=indent, scale=scale)

        for i in range(0, 3):
            lines[3 * i].write(begin_time=t0, transition_time=1)
            spheres[i].grow(begin_time=t0)
            t0 += 1.5

        segments = [
            Cylinder.from_start_to_end(
                start=z2loc(coords, 0),
                end=z2loc(coords, z1),
                name='3.3',
                color='drawing',
                thickness=0.5,
                label_rotation=[np.pi/2,-np.pi/2,0]
            ),
            Cylinder.from_start_to_end(
                end=z2loc(coords, 0),
                start=z2loc(coords, z2),
                name='1.4',
                color='example',
                thickness=0.5,
                label_rotation=[np.pi / 2, -np.pi/2, 0]
            ),
            Cylinder.from_start_to_end(
                start=z2loc(coords, 0),
                end=z2loc(coords, z1),
                name='4.5',
                color='important',
                thickness=0.6,
            )
        ]
        coords.add_objects(segments)

        arcs = [
            CircleArc(center=z2loc(coords, 0), radius=1, start_angle=0, end_angle=-np.arctan(1 / 1) + np.pi,
                      color='example', name=r'arg_z2_1', mode='XZ', thickness=0.5),
            CircleArc(center=z2loc(coords, 0), radius=1.25, start_angle=-np.arctan(1/3), end_angle=-np.arctan(1 / 3)+np.pi-np.arctan(1),
                      color='example', name=r'mul_arg_z2', mode='XZ', thickness=0.5),
            CircleArc(center=z2loc(coords, 0), radius=1, start_angle=0, end_angle=-np.arctan(1 / 1) + np.pi,
                      color='example', name=r'arg_z2_2', mode='XZ', thickness=0.5),
            CircleArc(center=z2loc(coords, 0), radius=1, start_angle=-np.arctan(1/3), end_angle=-np.arctan(1/3)-np.pi+np.arctan(1),
                      color='example', name=r'div_arg_z2', mode='XZ', thickness=0.5),
        ]
        coords.add_objects(arcs)

        # lines[1].write(begin_time=t0)
        segments[0].grow(modus='from_start', begin_time=t0)
        # segments[0].write_name_as_label(modus='right', offset=[4, 0, 1], begin_time=t0)
        t0 += 1.1
        # arcs[0].grow(begin_time=t0)
        # arcs[0].write_name_as_label(modus='center', begin_time=t0, offset=[-2.3, 0, 0])
        # lines[2].write(begin_time=t0)
        t0 += 1.1

        # lines[4].write(begin_time=t0)
        segments[1].grow(modus='from_end', begin_time=t0)
        # segments[1].write_name_as_label(modus='right', offset=[4, 0, -1], begin_time=t0)
        t0 += 1.1
        # arcs[1].grow(begin_time=t0)
        # arcs[1].write_name_as_label(modus='center', begin_time=t0, offset=[-0.9, 0, -0.2])
        # lines[5].write(begin_time=t0)
        t0 += 1.1

        composite = BObject(children=[segments[2], spheres[3]], name='Composite')
        composite.appear(begin_time=t0, transition_time=0)
        segments[2].grow(modus='from_start', begin_time=t0)
        spheres[3].grow(begin_time=t0)
        t0 += 2.1
        # lines[7].write(begin_time=t0)
        segments[2].rescale(rescale=[1, 1, 1.41], begin_time=t0)
        spheres[3].move_to(target_location=z2loc(coords, z1 * 1.41), begin_time=t0)
        # segments[2].write_name_as_label(modus='right', offset=[4, 0, 1], begin_time=t0 + 0.5)
        t0 += 1.1
        # lines[8].write(begin_time=t0)
        coords.add_object(composite)
        composite.rotate(rotation_euler=[0, -np.pi / 180 * 135, 0], begin_time=t0, transition_time=2)
        t0 += 2.1

        arcs[0].grow(begin_time=t0)
        print("arc 1 ",t0*FRAME_RATE)
        t0+=2
        arcs[1].grow(begin_time=t0)
        t0 += 2

        lines2 = [
            SimpleTexBObject(r'z_1=' + z2str(z1), color='drawing'),
            SimpleTexBObject(r'|z_1|=\sqrt{10}\approx3.3', color='drawing'),
            SimpleTexBObject(r'\arg(z_1)=-\arctan\left({1\over3}\right)\approx-18^\circ', color='drawing'),
            SimpleTexBObject(r'z_2=' + z2str(z2), color='example'),
            SimpleTexBObject(r'|z_1|=\sqrt{2}\approx1.4', color='example'),
            SimpleTexBObject(r'\arg(z_2)=-\arctan\left(1\right)+180^\circ=135^\circ', color='example'),
            SimpleTexBObject(r'{z_1 \over z_2}=' + z2str(z1 / z2), color='important'),
            SimpleTexBObject(r'\left|{z_1 \over z_2}\right|={\sqrt{10}\over \sqrt{2}}\approx 2.2', color='important'),
            SimpleTexBObject(r'\arg\left({z_1\over z_2}\right)\approx -18^\circ-135^\circ=-153^\circ', color='important'),
        ]

        for i in range(0,9,3):
            line = lines2[i]
            if i % 3 == 0:
                scale = 0.7
                indent = 1
            else:
                scale = 0.5
                indent = 1.5
            display.add_text_in_back(line, line=i + 1 + int(i / 3) * 1+0.75*int(i/7)*(i-6), indent=indent, scale=scale)
            if i<6:
                line.write(begin_time=t0,transition_time=0)

        display.turn(begin_time=t0)
        title2 = SimpleTexBObject(r"\text{Multiplication}", color='text', aligned="center", name='Title')
        display.set_title_back(title2)
        title2.write(begin_time=t0, transition_time=0)

        title3 = SimpleTexBObject(r"\text{Division}", color='text', aligned="center", name='Title')
        display.set_title_back(title3)
        title2.replace(title3,begin_time=t0+0.5, transition_time=1)
        composite.disappear(begin_time=t0)
        arcs[0].disappear(begin_time=t0)
        arcs[1].disappear(begin_time=t0)
        spheres[2].disappear(begin_time=t0)

        t0+=2

        # show result
        lines2[6].write(begin_time=t0)
        div_sphere=Sphere(0.25, location=coords.coords2location(z2p(z1/z2)), smooth=2, name='Sz1divz2', color='important')
        div_sphere.grow(begin_time=t0)
        div_mover= Sphere(0.24, location=coords.coords2location(z2p(z1)),smooth=2, name='div_mover',color='important')
        div_mover.grow(begin_time=t0)

        div_segment=Cylinder.from_start_to_end(
            start=z2loc(coords, 0),
            end=z2loc(coords, z1),
            name='2.2',
            color='important',
            thickness=0.6,
            label_rotation=[np.pi / 2, np.pi/2, 0]
        )
        div_segment.grow(modus='from_start', begin_time=t0)
        # div_arc =  CircleArc(center=z2loc(coords, 0), radius=3, start_angle=0, end_angle=np.angle(z1 / z2),
        #               color='important', name=r'-153^\circ', mode='XZ', thickness=0.5)

        div_composite= BObject(children=[div_segment, div_mover], name='Composite')
        div_composite.appear(begin_time=t0, transition_time=0)
        coords.add_objects([div_sphere, div_composite])

        t0+=1.5

        div_segment.rescale(rescale=[1, 1, 1/1.41], begin_time=t0)
        div_mover.move_to(target_location=z2loc(coords, z1/1.41), begin_time=t0)
        # lines2[7].write(begin_time=t0)
        # div_segment.write_name_as_label(offset=[-5, 0, 2], begin_time=t0+0.75, transition_time=0.5)
        # div_segment.label.rescale(rescale=[1,1,1.4],begin_time=t0)
        t0+=1.5

        # lines2[8].write(begin_time=t0)
        div_composite.rotate(rotation_euler=[0, np.pi / 180 * 135, 0], begin_time=t0, transition_time=2)
        t0+=2.5
        arcs[2].grow(begin_time=t0)
        t0+=1
        arcs[3].grow(begin_time=t0)
        # div_arc.grow(begin_time=t0)
        # div_arc.write_name_as_label(modus='center', offset=[-2.5, 0, 0], begin_time=t0 + 0.5)
        t0+=3

        print(t0*FRAME_RATE)

    def rotation(self):
        cues = self.sub_scenes['rotation']
        t0 = 0.5  # cues['start']

        z1 = 1 + 1j

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        display = Display(number_of_lines=18, columns=1, name='Display')
        title = SimpleTexBObject(r"\text{Rotations}", color='text', aligned="center", name='Title')
        display.set_title(title)

        display.appear(begin_time=t0, transition_time=1)
        title.write(begin_time=t0, transition_time=1)

        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-3, 3], [-3, 3]], radii=[0.03, 0.03],
                                  all_n_tics=[6, 6],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-3, 3.1, 1), np.arange(-3, 3.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[0, 0, 0],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=2)
        # draw grid lines
        # coords.draw_grid_lines(colors=['drawing', 'drawing'], begin_time=t0, transition_time=2, sub_grid=5)
        t0 += 3

        colors = flatten([['drawing'], ['example'] * 2, ['drawing'] * 6, ['example'] * 2, ['drawing'], ['example'] * 4])
        colors2 = flatten(
            [['drawing'], ['important'] * 2, ['drawing'] * 3, ['important'] * 2, ['drawing'] * 5, ['important'],
             ['text'], ['drawing'] * 2, ['important'], ['text'] * 9, ['drawing'], ['important'] * 2, ['text'] * 2,
             ['important'], ['drawing'], ['important']])
        lines = [
            SimpleTexBObject(r'z=' + z2str(z1), color='drawing'),
            SimpleTexBObject(r'\text{Remember: }', color='text'),
            SimpleTexBObject(r'\arg(i)=90^\circ', color='text'),
            SimpleTexBObject(r'|i|=1', color='text'),
            SimpleTexBObject(r'\text{counter-clockwise: }', color='drawing'),
            SimpleTexBObject(r'z\cdot i=(' + z2str(z1) + ')\cdot i=' + z2str(z1 * 1j), color=colors),
            SimpleTexBObject(r'\text{clockwise: }', color='drawing'),
            SimpleTexBObject(
                r'{z\over i}={' + z2str(z1) + '\over i}={(' + z2str(z1) + ')\cdot(-i)\over i\cdot(-i)}={' + z2str(
                    z1 / 1j) + '\over 1}=' + z2str(z1 / 1j), color=colors2),
        ]

        spheres = [
            Sphere(0.25, location=coords.coords2location(z2p(z1)), smooth=2, name='1+i', color='drawing'),
            Sphere(0.25, location=coords.coords2location(z2p(z1 * 1j)), smooth=2, name='-1+i', color='example'),
            Sphere(0.25, location=coords.coords2location(z2p(z1 / 1j)), smooth=2, name='1-i', color='important'),
        ]
        coords.add_objects(spheres)

        indents = [1, 1.5, 1.5, 1.5, 1, 1, 1, 1]
        offsets = [1, 2, 2, 2, 3, 3, 4, 4.5]
        scales = [0.7, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.55]
        for i, line in enumerate(lines):
            display.add_text_in(line, line=i + offsets[i], indent=indents[i], scale=scales[i])

        segments = [
            Cylinder.from_start_to_end(
                start=z2loc(coords, 0),
                end=z2loc(coords, z1),
                color='drawing',
                thickness=0.5,
                loop_cuts=3,
            ),
        ]
        coords.add_objects(segments)

        arcs = [
            CircleArc(center=z2loc(coords, 0), radius=1, start_angle=np.pi / 4, end_angle=3 * np.pi / 4,
                      color='example', name=r'\cdot', mode='XZ', thickness=0.5),
            CircleArc(center=z2loc(coords, 0), radius=1, start_angle=np.pi / 4, end_angle=-np.pi / 4,
                      color='important', name=r'\cdot\,\!', mode='XZ', thickness=0.5)
        ]
        coords.add_objects(arcs)

        # animation
        removables = []
        lines[0].write(begin_time=t0)
        spheres[0].grow(begin_time=t0 + 0.5)
        segments[0].grow(modus='from_start', begin_time=t0 + 1)
        spheres[0].write_name_as_label(modus='up_right', begin_time=t0 + 1)
        t0 += 2.5
        removables.append(spheres[0])
        removables.append(segments[0])

        for i in range(1, 4):
            lines[i].write(begin_time=t0, transition_time=0.5)
            t0 += 1

        lines[4].write(begin_time=t0)
        t0 += 1.5
        lines[5].write(begin_time=t0)
        t0 += 1.5

        mover_sphere = Sphere(0.24, location=coords.coords2location(z2p(z1)), smooth=2, name='MoverSphere',
                              color='drawing')
        mover_line = Cylinder.from_start_to_end(start=z2loc(coords, 0), end=z2loc(coords, z1), name='MoverLine',
                                                color='drawing', thickness=0.49, loop_cuts=3, )
        mover = BObject(children=[
            mover_sphere,
            mover_line
        ], name='Mover')
        coords.add_object(mover)
        mover.appear(begin_time=t0)
        mover_sphere.appear(begin_time=t0)
        mover_line.appear(begin_time=t0)
        removables.append(mover_sphere)
        removables.append(mover_line)

        mover.rotate(rotation_euler=[0, -np.pi / 2, 0], begin_time=t0)
        mover_sphere.change_color(new_color='example', begin_time=t0, )
        mover_line.change_color(new_color='example', begin_time=t0, )
        arcs[0].grow(begin_time=t0)
        t0 += 1.5
        arcs[0].write_name_as_label(modus='center', begin_time=t0, thickness=3, text_size=3, offset=[-0.7, 0, -0.7])
        spheres[1].grow(begin_time=t0)
        spheres[1].write_name_as_label(modus='up_left', begin_time=t0)
        removables.append(spheres[1])
        # removables.append(arcs[0])

        t0 += 1.5

        lines[6].write(begin_time=t0)
        t0 += 1.5
        lines[7].write(begin_time=t0, transition_time=3)
        t0 += 3.5

        mover_sphere = Sphere(0.24, location=coords.coords2location(z2p(z1)), smooth=2, name='MoverSphere2',
                              color='drawing')
        mover_line = Cylinder.from_start_to_end(start=z2loc(coords, 0), end=z2loc(coords, z1), name='MoverLine2',
                                                color='drawing', thickness=0.49, loop_cuts=3, )
        mover = BObject(children=[
            mover_sphere,
            mover_line
        ], name='Mover')
        coords.add_object(mover)
        mover.appear(begin_time=t0)
        mover_sphere.appear(begin_time=t0)
        mover_line.appear(begin_time=t0)
        removables.append(mover_sphere)
        removables.append(mover_line)
        mover.rotate(rotation_euler=[0, np.pi / 2, 0], begin_time=t0)
        mover_sphere.change_color(new_color='important', begin_time=t0, )
        mover_line.change_color(new_color='important', begin_time=t0, )
        arcs[1].grow(begin_time=t0)
        removables.append(spheres[2])
        # removables.append(arcs[1])
        t0 += 1.5
        arcs[1].write_name_as_label(modus='center', begin_time=t0, thickness=3, text_size=3, offset=[-0.7, 0, -0.7])
        spheres[2].grow(begin_time=t0)
        spheres[2].write_name_as_label(modus='down_right', begin_time=t0)
        t0 += 1.5

        for sphere in spheres:
            sphere.label_disappear(begin_time=t0, transition_time=0.5)
        for arc in arcs:
            arc.label_disappear(begin_time=t0, transition_time=0.5)
            arc.disappear(begin_time=t0, transition_time=0.5)

        explosion = Explosion(removables)
        explosion.set_wind_and_turbulence(wind_location=[6, 0, -5], turbulence_location=[0, 0, 0],
                                          rotation_euler=[0, -np.pi / 4, 0], wind_strength=1.5, turbulence_strength=10)
        explosion.explode(begin_time=t0, transition_time=2)

        display.rotate(rotation_euler=[np.pi / 2, 0, np.pi - np.pi / 15], begin_time=t0)

        title2 = SimpleTexBObject(r"\text{Rotations}", color='text', aligned="center", name='Title')
        display.set_title_back(title2)
        title2.write(begin_time=t0, transition_time=1)
        t0 += 3

        # rotations around an arbitrary position
        zr = 2j

        colors2 = flatten([['text'] * 14, ['example'] * 2, ['text']])
        lines2 = [
            SimpleTexBObject(z2str(z1), color='drawing'),
            SimpleTexBObject(r'\text{Rotation around }' + '2i' + r'\text{?}', color=colors2),
            SimpleTexBObject(r'\text{Recipe:}', color='text'),
            SimpleTexBObject(r'\text{Subtract } 2i', color='example'),
            SimpleTexBObject(r'(1+i)\mapsto (1+i)-2i=(1-i)', color='drawing'),
            SimpleTexBObject(r'\text{Multiply with } i', color='important'),
            SimpleTexBObject(r'(1-i)\mapsto (1-i)\cdot i=(1+i)', color='drawing'),
            SimpleTexBObject(r'\text{Add } 2i', color='example'),
            SimpleTexBObject(r'(1+i)\mapsto (1+i)+2i= 1+3i', color='drawing'),
        ]

        indents = [1, 1.5, 1.5, 2, 1, 2, 1, 2, 1]
        offsets = [1, 2, 3, 4, 4, 4, 4, 4, 4]
        scales = [0.7, 0.7, 0.7, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5]
        for i, line in enumerate(lines2):
            display.add_text_in_back(line, line=i + offsets[i], indent=indents[i])

        spheres = [
            Sphere(0.25, location=coords.coords2location(z2p(z1)), smooth=2, name='1+i', color='drawing'),
            Sphere(0.25, location=coords.coords2location(z2p(zr)), smooth=2, name='2i', color='example'),
            # Sphere(0.25, location=coords.coords2location(z2p((z1-zr)*i+zr)), smooth=2, name='1-i', color='important'),
        ]
        coords.add_objects(spheres)

        segments = [
            Cylinder.from_start_to_end(
                start=z2loc(coords, zr),
                end=z2loc(coords, z1),
                color='drawing',
                thickness=0.5,
                loop_cuts=3,
            ),
        ]
        coords.add_objects(segments)

        # animations
        lines2[0].write(begin_time=t0)
        t0 += 1.1
        spheres[0].grow(begin_time=t0)
        spheres[0].write_name_as_label(modus='up_right', begin_time=t0 + 0.5)
        t0 += 1.1
        lines2[1].write(begin_time=t0)
        t0 += 1.1
        spheres[1].grow(begin_time=t0)
        spheres[1].write_name_as_label(modus='up_left', begin_time=t0 + 0.5)
        segments[0].grow(modus='from_start', begin_time=t0 + 0.5)
        t0 += 1.6

        mover_sphere = Sphere(0.24, label_rotation=[np.pi / 2, np.pi / 2, 0],
                              location=coords.coords2location(z2p(z1 - zr)), smooth=2, name='?',
                              color='drawing')
        mover_line = Cylinder.from_start_to_end(start=z2loc(coords, 0), end=z2loc(coords, z1 - zr), name='MoverLine2',
                                                color='drawing', thickness=0.49, loop_cuts=3, )

        mover_sphere.appear(begin_time=t0)
        mover_line.appear(begin_time=t0)
        mover = BObject(children=[
            mover_sphere,
            mover_line
        ], location=z2loc(coords, zr), name='Mover')
        mover.appear(begin_time=t0)
        coords.add_object(mover)

        t0 += 1

        mover.rotate(rotation_euler=[0, -np.pi / 2, 0], begin_time=t0)
        mover_sphere.change_color(new_color='important', begin_time=t0)
        mover_line.change_color(new_color='important', begin_time=t0)
        arc = CircleArc(center=z2loc(coords, 2j), radius=1, start_angle=-np.pi / 4, end_angle=np.pi / 4,
                        color='important', name=r'\,\!\cdot', mode='XZ', thickness=0.5)
        arc.grow(begin_time=t0)
        arc.write_name_as_label(modus='center', offset=[-.8, 0, -2.6], begin_time=t0 + 0.5, thickness=3, text_size=3)
        mover_sphere.write_name_as_label(modus='up_right', begin_time=t0)
        t0 += 2

        mover.rotate(rotation_euler=[0, 0, 0], begin_time=t0)
        arc.shrink(begin_time=t0)
        arc.hide(begin_time=t0 + 0.1)
        mover_sphere.label_disappear(begin_time=t0, transition_time=0.5)
        t0 += 1.5
        lines2[2].write(begin_time=t0)
        t0 += 1.5
        lines2[3].write(begin_time=t0)
        t0 += 1.5

        arrows = [
            PArrow(start=z2loc(coords, zr), end=z2loc(coords, (zr - zr)), name='Arrow1', color='example'),
            PArrow(start=z2loc(coords, zr), end=z2loc(coords, (zr - zr)), name='Arrow2', color='example'),
        ]

        coords.add_objects(arrows)

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=2)
        t0 += 3

        end_positions = [z2loc(coords, (z1 - zr)), z2loc(coords, (zr - zr))]

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]
        coords.add_objects(containers)

        for container in containers:
            container.appear(begin_time=t0)

        for sphere in spheres:
            sphere.move(direction=z2loc(coords, -2j), begin_time=t0)
        segments[0].move(direction=z2loc(coords, -2j), begin_time=t0)
        mover.move(direction=z2loc(coords, -2j), begin_time=t0)
        spheres[0].label_disappear(begin_time=t0)
        spheres[1].label_disappear(begin_time=t0)

        for container in containers:
            container.shrink(begin_time=t0)
        lines2[4].write(begin_time=t0)
        t0 += 1
        sphere_new2 = Sphere(0.24, location=z2loc(coords, (z1 - 2j)), name='1-i', color='drawing')
        sphere_new2.grow(begin_time=t0)
        coords.add_object(sphere_new2)
        sphere_new2.write_name_as_label(modus='down_right', begin_time=t0 + 0.5)
        t0 += 1.5
        lines2[5].write(begin_time=t0)
        t0 += 1.5

        mover.rotate(rotation_euler=[0, -np.pi / 2, 0], begin_time=t0)
        mover_sphere.change_color(new_color='important', begin_time=t0)
        mover_line.change_color(new_color='important', begin_time=t0)
        arc2 = CircleArc(center=z2loc(coords, 0), radius=1, start_angle=-np.pi / 4, end_angle=np.pi / 4,
                         color='important', name=r'\cdot\!\,', mode='XZ', thickness=0.5)
        arc2.grow(begin_time=t0)
        arc2.write_name_as_label(modus='center', offset=[-.8, 0, 0.7], begin_time=t0 + 0.5, thickness=3, text_size=3)
        lines2[6].write(begin_time=t0)
        t0 += 1.5
        sphere_new = Sphere(0.24, location=z2loc(coords, (z1 - 2j) * 1j), name='(1-i)\cdot i', color='important')
        sphere_new.grow(begin_time=t0)
        coords.add_object(sphere_new)
        sphere_new.write_name_as_label(modus='right', begin_time=t0 + 0.5, aligned='left', offset=[0.3, 0, 0])
        lines2[7].write(begin_time=t0)
        t0 += 1.5

        arrows2 = [
            PArrow(start=z2loc(coords, -2j), end=z2loc(coords, 0), name='Arrow3', color='example'),
            PArrow(start=z2loc(coords, -2j), end=z2loc(coords, 0), name='Arrow4', color='example'),
            PArrow(start=z2loc(coords, -2j), end=z2loc(coords, 0), name='Arrow5', color='example'),
        ]

        coords.add_objects(arrows2)

        for arrow in arrows2:
            arrow.grow(begin_time=t0, transition_time=2)
        t0 += 3

        end_positions2 = [z2loc(coords, zr), z2loc(coords, z1), z2loc(coords, (z1 - 2j) * 1j + 2j)]
        containers2 = [BObject(children=[arrows2[i]], location=end_positions2[i]) for i in range(0, len(arrows2))]
        coords.add_objects(containers2)

        for container in containers2:
            container.appear(begin_time=t0)

        for sphere in spheres:
            sphere.move(direction=z2loc(coords, 2j), begin_time=t0)
        segments[0].move(direction=z2loc(coords, 2j), begin_time=t0)
        mover.move(direction=z2loc(coords, 2j), begin_time=t0)
        arc2.move(direction=z2loc(coords, 2j), begin_time=t0)
        sphere_new.move(direction=z2loc(coords, 2j), begin_time=t0)
        sphere_new.label_disappear(begin_time=t0)
        sphere_new.disappear(begin_time=t0)
        sphere_new2.move(direction=z2loc(coords, 2j), begin_time=t0)
        sphere_new2.label_disappear(begin_time=t0)
        sphere_new2.disappear(begin_time=t0)

        for container in containers2:
            container.shrink(begin_time=t0)

        lines2[8].write(begin_time=t0)
        t0 += 2

        sphere_new3 = Sphere(0.24, location=z2loc(coords, (z1 - 2j) * 1j + 2j), name='1+3i',
                             color='important')
        sphere_new3.grow(begin_time=t0)
        coords.add_object(sphere_new3)
        sphere_new3.write_name_as_label(modus='right', aligned='left', offset=[0.3, 0, 0], begin_time=t0 + 0.5)

        print(t0)

    def construction(self, t0, title, durations=[0, 0, 0, 0], parts=[True, True, True], locs=[0, 1, 0.75 + 1.5j, 0.75],
                     grid_lines=False, text=True, details=0, prefix=''):

        # setup
        self.construction_counter+=1 # make a unique counter for each new construction
        duration0 = durations[0]
        dt = duration0 / 2

        duration1 = durations[1]
        n1 = 6
        dt1 = 0.8 * duration1 / n1  # action time
        sep1 = 0.2 * duration1 / n1  # pause

        duration2 = durations[2]
        n2 = 4
        dt2 = 0.8 * duration2 / n2
        sep2 = 0.2 * duration2 / n2

        duration3 = durations[3]
        n3 = 3
        dt3 = 0.8 * duration3 / n3
        sep3 = 0.2 * duration3 / n3

        # coord
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 1], [-1, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[3, 3],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2., 1.1, 1), np.arange(-1, 2.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[2, 0, -2],
                                  name=prefix + 'ComplexPlane')
        var_list = [coords]

        if text:
            # display0
            display0 = Display(scales=[4, 2.5], location=[12.5, 0, 2.5], number_of_lines=9, name=prefix + 'DisplayOben')
            display0.appear(begin_time=t0, transition_time=dt)

            # display 1
            display1 = Display(number_of_lines=10, columns=1, location=[12.5, 0, -2.5], scales=[4, 2.5],
                               name=prefix + 'DisplayUnten')
            display1.appear(begin_time=t0, transition_time=0)

            var_list.append(display0)
            var_list.append(display1)

            display0.set_title(title)
            title.write(begin_time=t0 + dt, transition_time=dt)

        coords.appear(begin_time=t0, transition_time=2 * dt)

        if grid_lines:
            coords.draw_grid_lines(colors=['drawing', 'drawing'], begin_time=t0 + dt,
                                   transition_time=dt, sub_grid=5, loop_cuts=details)
        t0 += duration0

        # string data
        colors = [
            flatten([['text'] * 3, ['drawing'] * 3, ['text']]),
            flatten([['text'] * 6, ['example'], ['text']]),
            flatten([['text'] * 8, ['drawing'], ['text'] * 6, ['drawing']]),
            flatten([['text'] * 3, ['joker'], ['text'] * 13, ['example']]),
            flatten([['text'] * 6, ['example'] * 13, ['text'] * 2, ['drawing'] * 3]),
            flatten([['text'] * 14, ['example']]),
            flatten([['text']]),
            flatten([['text'] * 15, ['important'], ['text'] * 2, ['drawing'] * 2]),
            flatten([['text']]),
            flatten([['important'] * 4, ['text'] * 3, ['important'] * 4, ['text']]),
            flatten([['text'] * 14, ['example'] * 4, ['text'] * 7, ['important']]),
            flatten([['text'] * 3, ['example'] * 13, ['text'] * 2, ['drawing']]),
            flatten([['text'] * 10, ['drawing'] * 3, ['text'] * 8, ['important'], ['text']]),
            flatten([['text'] * 26, ['joker']]),
        ]
        if text:
            lines = [
                SimpleTexBObject(r"\text{Let $\overline{AB}$ be a line segment}", aligned='left', color=colors[0],name="con_line_00"),
                SimpleTexBObject(r"\text{and let $T$ be a point on it}", aligned='left', color=colors[1],name="con_line_01"),
                SimpleTexBObject(r"\text{closer to $B$ than to $A$.}", aligned='left', color=colors[2],name="con_line_02"),
                SimpleTexBObject(r"\text{Let $C$ be a point on the line}", aligned='left', color=colors[3],name="con_line_03"),
                SimpleTexBObject(r"\text{that is perpendicular to $\overline{AB}$}", aligned='left', color=colors[4],name="con_line_04"),
                SimpleTexBObject(r"\text{and goes through $T$.}", aligned='left', color=colors[5],name="con_line_05"),
                SimpleTexBObject(r"\text{1.) Show that there is }", aligned='left', color=colors[6],name="con_line_06"),
                SimpleTexBObject(r"\text{exactly one point $D$ on $\overline{AC}$}", aligned='left', color=colors[7],name="con_line_07"),
                SimpleTexBObject(r"\text{such that the angles}", aligned='left', color=colors[8],name="con_line_08"),
                SimpleTexBObject(r"\text{$\angle CBD$ and $\angle BAC$ are the same.}", aligned='left',
                                 color=colors[9],name="con_line_08"),
                SimpleTexBObject(r"\text{2.) Show that the line through $D$}", aligned='left', color=colors[10],name="con_line_09"),
                SimpleTexBObject(r"\text{and perpendicular to $\overline{AC}$}", aligned='left', color=colors[11],name="con_line_10"),
                SimpleTexBObject(r"\text{intersects $\overline{AB}$ in a point $E$ that}", aligned='left',
                                 color=colors[12],name="con_line_11"),
                SimpleTexBObject(r"\text{does not depend on the choice of $C$.}", aligned='left', color=colors[13],name="con_line_12"),
            ]
            indents = flatten([[0.5] * 7, [1] * 3, [0.5], [1] * 4])
            var_list.append(lines)

        # part 1
        if parts[0]:
            if text:
                for i in range(0, 6):
                    line = lines[i]
                    display0.add_text_in(line, line=1 + i, indent=indents[i])
                    line.write(begin_time=t0 + i * (dt1 + sep1), transition_time=dt1)

            # objects in coordinate system
            [a, b, c, t] = locs
            a_loc = coords.coords2location(z2p(a))
            b_loc = coords.coords2location(z2p(b))
            t_loc = coords.coords2location(z2p(t))
            c_loc = coords.coords2location(z2p(c))

            s_a = Sphere(0.25, location=a_loc, name=prefix + 'A=0')
            s_a.grow(begin_time=t0, transition_time=dt1)

            s_b = Sphere(0.25, location=b_loc, name=prefix + 'B=1')
            s_b.grow(begin_time=t0, transition_time=dt1)

            l_ab = Cylinder.from_start_to_end(start=a_loc, end=b_loc, thickness=0.5, name=prefix + 'L_ab')
            l_ab.grow(begin_time=t0, modus='from_start', transition_time=dt1)

            t0 += dt1 + sep1

            s_t = Sphere(0.25, location=t_loc, name=prefix + 'T=x', color='example')
            s_t.grow(begin_time=t0, transition_time=dt1)

            t0 += 2 * (dt1 + sep1)

            offset = coords.coords2location([0, 0.2])
            s_c = Sphere(0.25, location=c_loc, name=prefix + 'C=x+y\,i', color='joker')
            s_c.grow(begin_time=t0, transition_time=dt1)

            t0 += dt1 + sep1

            l_tc = Cylinder.from_start_to_end(start=t_loc - offset, end=c_loc + offset,
                                              color='example', thickness=0.5, loop_cuts=details, name=prefix + 'L_tc')
            l_tc.grow(begin_time=t0, transition_time=2 * dt1)
            ra = RightAngle(location=t_loc, radius=0.65, thickness=0.4, mode='XZ', color='example',
                            name='RA_C' + str(self.construction_counter))
            ra.appear(begin_time=t0, transition_time=2 * dt1)

            t0 += 2 * (dt1 + sep1)

            vars_part1 = [s_a, s_b, l_ab, s_t, l_tc, s_c,ra]
            coords.add_objects(vars_part1)
            vars_part1+=[a_loc,b_loc,c_loc,t_loc]
            var_list += vars_part1

        # first problem
        if parts[1]:
            if text:
                for i in range(0, 4):
                    line = lines[i + 6]
                    display1.add_text_in(line, line=i, indent=indents[i + 6])
                    line.write(begin_time=t0 + i * (dt2 + sep2), transition_time=dt2)

            [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)
            s_d = Sphere(0.25, location=d_loc, color='important', name=prefix + 'D', smooth=2)
            s_d.appear(begin_time=t0, transition_time=dt2)
            t0 += dt2 + sep2

            l_ac = Cylinder.from_start_to_end(start=a_loc, end=c_loc, color='drawing', thickness=0.5,
                                              name=prefix + 'L_ac')
            l_ac.grow(begin_time=t0, transition_time=dt2, modus='from_start')
            t0 += dt2 + sep2

            l_bc = Cylinder.from_start_to_end(start=b_loc, end=c_loc, color='text', thickness=0.25,
                                              name=prefix + 'L_bc')
            l_bd = Cylinder.from_start_to_end(start=b_loc, end=d_loc, color='text', thickness=0.25,
                                              name=prefix + 'L_bd')

            l_bc.grow(begin_time=t0, transition_time=dt2, modus='from_start')
            l_bd.grow(begin_time=t0, transition_time=dt2, modus='from_start')

            t0 += dt2 + sep2

            # define the arc with the full possible range to be able to dynamically adjust it
            bac = CircleArc(center=a_loc, radius=0.5, start_angle=0, end_angle=np.pi / 2, color='important',
                            name=prefix + 'Arc_bac', mode='XZ', thickness=0.5)
            bac.appear(begin_time=t0, transition_time=dt2)
            bac.extend_to(2 * alpha / np.pi, begin_time=t0, transition_time=0)

            cbd = CircleArc(center=b_loc, radius=0.5, start_angle=0, end_angle=np.pi, color='important',
                            name=prefix + 'Arc_cbd', mode='XZ', thickness=0.5)
            cbd.appear(begin_time=t0, transition_time=dt2)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t0, transition_time=0)
            t0 += dt2 + sep2

            vars_part2 = [s_d, l_ac, l_bc, l_bd, bac, cbd]
            coords.add_objects(vars_part2)
            var_list += vars_part2

        # second problem
        if parts[2]:
            if text:
                for i in range(0, 4):
                    line = lines[i + 10]
                    display1.add_text_in(line, line=i + 4, indent=indents[i + 10])
                    line.write(begin_time=t0 + i * (dt3 + sep3), transition_time=dt3)

            [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)

            l_de = Cylinder.from_start_to_end(start=e_loc, end=d_loc, color='example', thickness=0.5)
            l_de.grow(begin_time=t0, transition_time=2 * dt3, modus='from_start')
            t0 += 2 * (dt3 + sep3)

            s_e = Sphere(0.25, location=e_loc, color='important', name=prefix + 'E', smooth=2)
            s_e.appear(begin_time=t0, transition_time=2 * dt3)
            ra2 = RightAngle(location=d_loc, rotation_euler=[0,np.pi/2- alpha, 0], radius=0.65, thickness=0.4,
                             mode='XZ', color='example', name='RA_D'+str(self.construction_counter))
            ra2.appear(begin_time=t0, transition_time=dt3)
            t0 += 2 * (dt3 + sep3)

            vars_part3 = [s_e, l_de,ra2]
            coords.add_objects(vars_part3)
            var_list += vars_part3

        # convert all variables in a dictionary
        dictionary = {}
        for var in var_list:
            dictionary[retrieve_name(var)[0]] = var
        return dictionary

    def solution1(self):
        cues = self.sub_scenes['solution1']
        t0 = 0  # cues['start']

        debug = False
        details = 3
        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        title0 = SimpleTexBObject(r"\text{Problem 3}", color='important', aligned='center')
        dictionary = self.construction(t0, title0, parts=[True, False, False], grid_lines=True, details=details)

        coords = dictionary['coords']
        display0 = dictionary['display0']
        display1 = dictionary['display1']
        s_a = dictionary['s_a']
        s_b = dictionary['s_b']
        s_c = dictionary['s_c']
        s_t = dictionary['s_t']
        l_tc = dictionary['l_tc']
        a_loc = dictionary['a_loc']
        b_loc = dictionary['b_loc']
        t_loc = dictionary['t_loc']
        c_loc = dictionary['c_loc']
        ra = dictionary['ra']

        title = SimpleTexBObject(r"\text{Without loss of generality}", aligned="center", name='Title', color='text')
        display1.set_title(title)
        title.write(begin_time=t0, transition_time=0)

        if not debug:
            lines2 = [
                SimpleTexBObject(r"A=0", aligned='left'),
                SimpleTexBObject(r"B=1", aligned='left'),
                SimpleTexBObject(r"T=x", aligned='left'),
                SimpleTexBObject(r"x\in (0.5,1]", aligned='left'),
                SimpleTexBObject(r"C=x+y\,i", aligned='left'),
                SimpleTexBObject(r"y>0", aligned='left')
            ]
            rows = [1, 2, 4, 4, 5, 5]
            indents = [1, 1, 1, 4, 1, 4]
            for text, row, indent in zip(lines2, rows, indents):
                display1.add_text_in(text, line=row, indent=indent)
                text.write(begin_time=t0, transition_time=0)

        s_a.write_name_as_label(letter_set=[0], begin_time=t0, modus='down_left', transition_time=0)
        s_b.write_name_as_label(letter_set=[0], begin_time=t0, modus='down_right', transition_time=0)
        s_c.write_name_as_label(begin_time=t0, modus='down_left', transition_time=0, offset=[-0.59, 0, 0])
        s_t.write_name_as_label(modus="down_left", transition_time=0, offset=[-0.15454, 0, 0])

        # complete labels
        t0 += 1
        s_a.write_name_as_label(letter_set=[1, 2], modus='down_left', begin_time=t0, transition_time=0.5)
        s_b.write_name_as_label(letter_set=[1, 2], modus='down_right', begin_time=t0, transition_time=0.5)

        # rotate for showing the question
        t0 += 1
        display1.rotate(rotation_euler=[np.pi / 2, 0, np.pi - np.pi / 15], begin_time=t0)
        title0.replace(SimpleTexBObject(r"\text{Solution -- Part 1}", color='important', aligned='center'),
                       begin_time=t0)

        # remove things
        if not debug:
            removables = flatten([coords.x_lines, coords.y_lines])
            removables.append(l_tc)
            ra.disappear(begin_time=t0)
            explosion = Explosion(removables)
            explosion.set_wind_and_turbulence(wind_location=[6, 0, -5], turbulence_location=[0, 0, 0],
                                              rotation_euler=[0, -np.pi / 4, 0], wind_strength=1.5,
                                              turbulence_strength=10)
            explosion.explode(begin_time=t0, transition_time=2)

            t0 += 2.5

        # write question

        if not debug:
            colors = [
                flatten([['text']]),
                flatten([['text'] * 15, ['important'], ['text'] * 2, ['drawing'] * 2]),
                flatten([['text']]),
                flatten([['important'] * 4, ['text'] * 3, ['important'] * 4, ['text']]),
            ]

            lines = [
                SimpleTexBObject(r"\text{1.) Show that there is }", aligned='left', color=colors[0]),
                SimpleTexBObject(r"\text{exactly one point $D$ on $\overline{AC}$}", aligned='left', color=colors[1]),
                SimpleTexBObject(r"\text{such that the angles}", aligned='left', color=colors[2]),
                SimpleTexBObject(r"\text{$\angle CBD$ and $\angle BAC$ are the same.}", aligned='left',
                                 color=colors[3]),
            ]

            indents = [0.5, 1, 1, 1]
            for i, line in enumerate(lines):
                display1.add_text_in_back(line, line=i - 1, indent=indents[i])
                line.write(begin_time=t0, transition_time=0.5)
                t0 += 0.6

        # continue construction

        l_ac = Cylinder.from_start_to_end(start=a_loc, end=c_loc,
                                          color='drawing', thickness=0.5, loop_cuts=details, name='lac')
        l_ac.grow(begin_time=t0, modus='from_start')
        t0 += 1.5

        ac = c_loc - a_loc
        ab = b_loc - a_loc
        angle = np.arccos(ac.dot(ab) / ac.length / ab.length)

        s_c_dash = Sphere(0.25, location=c_loc - b_loc, color='joker', name="Sc", label_rotation=[np.pi / 2, angle, 0])
        s_c_dash.grow(begin_time=t0)

        l_bc = Cylinder.from_start_to_end(start=Vector(), end=c_loc - b_loc,
                                          color='drawing', thickness=0.5, loop_cuts=details, name='lbc')
        l_bc.grow(begin_time=t0, modus='from_start')

        l_bc2 = Cylinder.from_start_to_end(start=b_loc, end=c_loc,
                                           color='drawing', thickness=0.25, loop_cuts=details, name='lbc2')
        l_bc2.grow(begin_time=t0, modus='from_start')

        rot_box = BObject(children=[s_c_dash, l_bc], location=b_loc)

        t0 += 1.5

        cb = c_loc - b_loc
        angle2 = np.arccos(cb.dot(Vector([1, 0, 0])) / cb.length)
        bac2 = CircleArc(center=b_loc, radius=1.5, start_angle=angle2, end_angle=angle2 + angle, color='important',
                         name=r'\angle BAC2',
                         mode='XZ', thickness=0.5)

        bac2.grow(begin_time=t0)
        bac2.write_name_as_label(modus='center', begin_time=t0, name=r"\angle CBD", offset=[-3.16, 0, -1.62])
        rot_box.appear(begin_time=t0)
        rot_box.rotate(rotation_euler=[0, -angle, 0], begin_time=t0)

        t0 += 1.5

        s_c_dash.write_name_as_label(begin_time=t0, modus='up_left', name=r'C^\prime')

        t0 += 1.5

        bac = CircleArc(center=a_loc, radius=1.5, start_angle=0, end_angle=angle, color='important', name=r'\angle BAC',
                        mode='XZ', thickness=0.5)

        bac.grow(begin_time=t0)
        bac.write_name_as_label(modus='center', begin_time=t0, name=r"\angle BAC", offset=[-1.6, 0, 0])
        t0 += 1.5

        zc = 0.75 + 1.5j
        zb = 1
        zcp = zc * (zc - zb) / np.abs(zc) + zb
        cp_loc = coords.coords2location([np.real(zcp), np.imag(zcp)])
        offset1 = (cp_loc - b_loc).normalized()
        offset2 = (c_loc - a_loc).normalized()
        line1 = Cylinder.from_start_to_end(start=b_loc - 2 * offset1, end=cp_loc + 2 * offset1, thickness=0.25,
                                           color='drawing')
        line2 = Cylinder.from_start_to_end(start=a_loc - 2 * offset2, end=c_loc + 2 * offset2, thickness=0.25,
                                           color='drawing')
        seg_ac = Cylinder.from_start_to_end(start=a_loc, end=c_loc , thickness=0.5,
                                           color='drawing')

        line1.grow(begin_time=t0)
        line2.grow(begin_time=t0)

        result = geometry(a_loc, b_loc, c_loc, t_loc)
        d_loc = result[0]
        s_d = Sphere(0.25, location=d_loc, color='important', name='D')
        s_d.grow(begin_time=t0 + 0.5)

        l_bc2.disappear(begin_time=t0)
        l_bc.disappear(begin_time=t0)
        l_ac.disappear(begin_time=t0)
        bac.disappear(begin_time=t0)
        bac2.disappear(begin_time=t0)

        s_d.write_name_as_label(modus='up', begin_time=t0 + 1)
        t0 += 2.5

        title0b = SimpleTexBObject(r"\text{Solution -- Part 1}", color='important', aligned='center')
        display0.set_title_back(title0b)
        title0b.write(begin_time=t0, transition_time=0)

        if not debug:
            for line in lines:
                line.disappear(begin_time=t0)

        display0.rotate(rotation_euler=[np.pi / 2, 0, -np.pi - np.pi / 15], begin_time=t0)
        line1.disappear(begin_time=t0)
        s_d.disappear(begin_time=t0)
        s_t.disappear(begin_time=t0)
        t0 += 1.5

        bac3 = CircleArc(center=a_loc, radius=1.5, start_angle=0, end_angle=angle, color='joker', name=r'\angle BAC3',
                         mode='XZ', thickness=0.5)

        bac3.grow(begin_time=t0)
        bac3.write_name_as_label(modus='center', begin_time=t0, name=r"\angle BAC", offset=[-1.6, 0, 0])
        t0 += 1.5
        bac3.disappear(begin_time=t0)
        line2.disappear(begin_time=t0)
        seg_ac.grow(begin_time=t0,transition_time=0.5)
        line2.appear(begin_time=t0+1,transition_time=0.5)
        seg_ac.disappear(begin_time=t0+1,transition_time=0.5)
        t0 += 1.5

        new_line = SimpleTexBObject(r'\text{Rotation of }C:', color='joker')
        display0.add_text_in_back(new_line, line=1, indent=0.5)
        new_line.write(begin_time=t0)

        coords.add_objects(l_ac, l_bc2, bac, bac2, bac3, rot_box, line1, line2, s_d,seg_ac)

    def solution1b(self):
        cues = self.sub_scenes['solution1b']
        t0 = 0  # cues['start']

        debug = False
        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # coord
        details = 3
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 1], [-1, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[3, 3],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2., 1.1, 1), np.arange(-1, 2.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[2, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)

        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18)
        title0 = SimpleTexBObject(r"\text{Solution -- Part 1}", color='important', aligned='center')
        display.set_title(title0)
        title0.write(begin_time=t0, transition_time=0)

        colors1 = flatten([['joker'] * 2, ['text'] * 10])

        lines = [
            SimpleTexBObject(r"\text{Rotation of }C:", aligned='left', color='joker', name='Line0'),
            SimpleTexBObject(r"C-B=x+yi-1", color='text', name='Line1'),
            SimpleTexBObject(r"(C-B)\cdot C=((x-1)+yi)\cdot(x+yi)", color='text', name='Line2'),
            SimpleTexBObject(r"=(x^2-x-y^2)+(xy-y+xy)i", color='text', name='Line3'),
            SimpleTexBObject(r"=(x^2-x-y^2)+(2xy-y)i", color='text', name='Line3b'),  # dummy line for replacement
            SimpleTexBObject(r"C^\prime=(C-B)\cdot C+B", color=colors1, name='Line5'),
            SimpleTexBObject(r"=(x^2-x-y^2+1)+(2xy-y)i", color='text', name='Line6'),
        ]
        indents = [0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        rows = [1, 2, 3, 4, 4, 5, 6]

        for line, row, indent in zip(lines, rows, indents):
            display.add_text_in(line, line=row, indent=indent)

        lines[2].align(lines[1], char_index=7, other_char_index=3)
        lines[3].align(lines[1], char_index=0, other_char_index=3)
        lines[4].align(lines[1], char_index=0, other_char_index=3)  # align dummy line
        lines[5].align(lines[1], char_index=2, other_char_index=3)
        lines[6].align(lines[1], char_index=0, other_char_index=3)
        lines[0].write(begin_time=t0)
        t0 += 1.5

        # objects in coordinate system
        a_loc = coords.coords2location([0, 0])
        s_a = Sphere(0.25, location=a_loc, name='A=0')
        s_a.grow(begin_time=t0, transition_time=0)
        s_a.write_name_as_label(begin_time=t0, modus='down_left', transition_time=0)

        b_loc = coords.coords2location([1, 0])
        s_b = Sphere(0.25, location=b_loc, name='B=1')
        s_b.grow(begin_time=t0, transition_time=0)
        s_b.write_name_as_label(begin_time=t0, modus='down_right', transition_time=0)

        l_ab = Cylinder.from_start_to_end(start=a_loc, end=b_loc, thickness=0.5)
        l_ab.grow(begin_time=t0, modus='from_start', transition_time=0)

        c_loc = coords.coords2location([0.75, 1.5])
        s_c = Sphere(0.25, location=c_loc, name='C=x+y\,i', color='joker')
        s_c.grow(begin_time=t0, transition_time=0)
        s_c.write_name_as_label(begin_time=t0, modus='down_left', offset=[-0.59, 0, 0])

        coords.add_objects([s_a, s_b, l_ab, s_c])
        t0 += 1.5

        # continue construction

        ac = c_loc - a_loc
        ab = b_loc - a_loc
        angle = np.arccos(ac.dot(ab) / ac.length / ab.length)

        s_c_dash = Sphere(0.25, location=c_loc - b_loc, color='joker', name="Sc", label_rotation=[np.pi / 2, angle, 0])
        s_c_dash.grow(begin_time=t0, transition_time=0)

        rot_box = BObject(children=[s_c_dash], location=b_loc)

        rot_box.appear(begin_time=t0)
        rot_box.rotate(rotation_euler=[0, -angle, 0], begin_time=t0, transition_time=0)
        s_c_dash.write_name_as_label(begin_time=t0, modus='up_left', name=r'C^\prime', transition_time=0)
        coords.add_object(rot_box)

        t0 += 0.5
        rot_box.rotate(rotation_euler=[0, 0, 0], begin_time=t0)
        t0 += 1.1
        rot_box.rotate(rotation_euler=[0, -angle, 0], begin_time=t0)

        t0 += 1.5

        # shift by B
        arrows = [
            PArrow(start=b_loc, end=Vector(), name='Arrow1', color='example', thickness=2),
            PArrow(start=b_loc, end=Vector(), name='Arrow2', color='example', thickness=2),
        ]

        coords.add_objects(arrows)

        lines[1].write(begin_time=t0)
        t0 += 1.5

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=1)

        t0 += 1.5

        end_positions = [c_loc - b_loc, Vector()]

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]
        coords.add_objects(containers)

        for container in containers:
            container.appear(begin_time=t0)

        s_m = Sphere(0.2, location=c_loc, color='text', name='C-B')
        s_m2 = Sphere(0.2, location=b_loc, color='text', name='B-B')

        spheres = [s_m2, s_m]
        modi = ['down_right', 'down_left']
        for modus, sphere in zip(modi, spheres):
            sphere.move(direction=-b_loc, begin_time=t0)
            sphere.write_name_as_label(modus=modus, begin_time=t0)

        for container in containers:
            container.shrink(begin_time=t0)

        t0 += 1.5

        lines[1].move_letters_to(lines[1], src_letter_indices=[5, 6, 7, 8, 9], target_letter_indices=[7, 8, 9, 5, 6],
                                 begin_time=t0, transition_time=2,
                                 offsets=[[0.1, 0, 0], [0.1, 0, 0], [0, 0, 0], [0, 0, 0], [-0.05, 0, 0]])

        t0 += 1.5
        # rotate
        lines[2].write(begin_time=t0, transition_time=2)
        t0 += 2.5

        ac = c_loc - a_loc
        ab = b_loc - a_loc
        angle = np.arccos(ac.dot(ab) / ac.length / ab.length)
        s_m3 = Sphere(0.2, location=c_loc - b_loc, color='text', name='(C-B)\cdot C',
                      label_rotation=[np.pi / 2, angle, 0])

        l_ac = Cylinder.from_start_to_end(start=a_loc, end=c_loc, thickness=0.5, color='joker', name='lac')
        l_ac.grow(begin_time=t0, modus='from_start')

        bac = CircleArc(center=a_loc, radius=1.5, start_angle=0, end_angle=angle, color='joker', name=r'\angle BAC3',
                        mode='XZ', thickness=0.5)

        bac.grow(begin_time=t0)
        bac.write_name_as_label(modus='center', begin_time=t0, name=r"\angle BAC", offset=[-1.6, 0, 0])
        t0 += 1.5

        r = (coords.coords2location([0, 1.5]) - coords.coords2location([0, 0])).length
        angle0 = np.angle(-0.3 + 1.5j)
        rotation = CircleArc(center=a_loc, radius=r, start_angle=angle0, end_angle=angle0 + angle, color='example',
                             name=r'Rotation',
                             mode='XZ', thickness=0.5)
        rotation.grow(begin_time=t0)

        t0 += 1.5

        s_m3.grow(begin_time=t0)

        rotation.shrink(begin_time=t0, inverted=True)
        rot_box2 = BObject(children=[s_m3], location=Vector(), name='RotationBox2')
        rot_box2.appear(begin_time=t0)
        rot_box2.rotate(rotation_euler=[0, -angle, 0], begin_time=t0)
        s_m3.write_name_as_label(modus='down_left', begin_time=t0)
        t0 += 1.5

        l_ac.disappear(begin_time=t0)
        bac.disappear(begin_time=t0)

        # simplify multiplication
        dt = 1
        dt_dis = 0.5
        sep = dt + 0.5

        lines[3].write(letter_set=[0, 1], begin_time=t0, transition_time=dt)
        t0 += sep
        set = [10, 20]
        lines[2].move_copy_to(lines[3], src_letter_indices=set, target_letter_indices=[2, 2], begin_time=t0,
                              transition_time=dt, new_color='example')
        lines[3].write(letter_set=[2, 3], begin_time=t0 + dt, transition_time=0)
        for i in range(len(set)):
            lines[2].copies_of_letters[-i].disappear(begin_time=t0 + dt, transition_time=dt_dis)
        t0 += sep

        t0 += sep
        set = [11, 12, 20]
        lines[2].move_copy_to(lines[3], src_letter_indices=set, target_letter_indices=[4, 5, 5], begin_time=t0,
                              transition_time=dt, new_color='example')
        lines[3].write(letter_set=[4, 5], begin_time=t0 + dt, transition_time=0)
        for i in range(len(set)):
            lines[2].copies_of_letters[-i].disappear(begin_time=t0 + dt, transition_time=dt_dis)
        t0 += sep

        set = [15, 16, 22, 23]
        lines[2].move_copy_to(lines[3], src_letter_indices=set, target_letter_indices=[7, 6, 7, 6], begin_time=t0,
                              transition_time=dt, new_color='example')
        lines[3].write(letter_set=[6, 7, 8], begin_time=t0 + dt, transition_time=0)
        for i in range(len(set)):
            lines[2].copies_of_letters[-i].disappear(begin_time=t0 + dt, transition_time=dt_dis)
        t0 += sep

        lines[3].write(letter_set=[9, 10, 11, 19], begin_time=t0, transition_time=dt)
        t0 += sep

        set = [10, 22, 23]
        lines[2].move_copy_to(lines[3], src_letter_indices=set, target_letter_indices=[12, 13, 20], begin_time=t0,
                              transition_time=dt, new_color='example')
        lines[3].write(letter_set=[12, 13, 20], begin_time=t0 + dt, transition_time=0)
        for i in range(len(set)):
            lines[2].copies_of_letters[-i].disappear(begin_time=t0 + dt, transition_time=dt_dis)
        t0 += sep

        set = [11, 12, 22, 23]

        lines[2].move_copy_to(lines[3], src_letter_indices=set, target_letter_indices=[14, 15, 15, 20], begin_time=t0,
                              transition_time=dt, new_color='example')
        lines[3].write(letter_set=[14, 15], begin_time=t0 + dt, transition_time=0)
        for i in range(len(set)):
            lines[2].copies_of_letters[-i].disappear(begin_time=t0 + dt, transition_time=dt_dis)
        t0 += sep

        set = [14, 15, 16, 20]

        lines[2].move_copy_to(lines[3], src_letter_indices=set, target_letter_indices=[16, 18, 20, 17], begin_time=t0,
                              transition_time=dt, new_color='example')
        lines[3].write(letter_set=[16, 17, 18], begin_time=t0 + dt, transition_time=0)
        for i in range(len(set)):
            lines[2].copies_of_letters[-i].disappear(begin_time=t0 + dt, transition_time=dt_dis)

        t0 += 2
        lines[3].replace(lines[4], src_letter_range=[11, 21], img_letter_range=[11, 19],
                         begin_time=t0, transition_time=2)
        t0 += 2.5

        # shift back

        arrows2 = [
            PArrow(end=Vector(), start=-b_loc, name='Arrow2', color='example', thickness=2),
            PArrow(end=Vector(), start=-b_loc, name='Arrow3', color='example', thickness=2),
            PArrow(end=Vector(), start=-b_loc, name='Arrow4', color='example', thickness=2),
        ]

        lines[5].write(begin_time=t0)
        t0 += 1.5

        for arrow in arrows2:
            arrow.grow(begin_time=t0)

        zc = 0.75 + 1.5j
        zb = 1
        zcp = zc * (zc - zb) / np.abs(zc) + zb
        cp_loc = coords.coords2location([np.real(zcp), np.imag(zcp)])

        s_m4 = Sphere(0.2, location=cp_loc - b_loc, name='S4', color='text')
        s_m4.grow(begin_time=t0, transition_time=0)
        s_m4.write_name_as_label(modus='down', begin_time=t0, transition_time=0, name=r'(C-B)\cdot C',
                                 offset=[0.18, 0, -0.14])
        s_m3.disappear(begin_time=t0)

        t0 += 1.5

        coords.add_objects(arrows2)

        end_positions2 = [c_loc, b_loc, cp_loc]

        containers2 = [BObject(children=[arrows2[i]], location=end_positions2[i]) for i in range(0, len(arrows2))]
        coords.add_objects(containers2)

        for container in containers2:
            container.appear(begin_time=t0)

        modi = ['down_right', 'down_left']
        for modus, sphere in zip(modi, spheres):
            sphere.move(direction=b_loc, begin_time=t0)
            sphere.label_disappear(begin_time=t0)

        for container in containers2:
            container.shrink(begin_time=t0)

        s_m4.move(direction=b_loc, begin_time=t0)
        s_m4.label.replace(SimpleTexBObject(r'(C-B)\cdot C+B', aligned='center', color='text'), begin_time=t0)

        t0 += 1.5

        lines[6].write(letter_set=[0], begin_time=t0, transition_time=dt)
        t0 += sep
        lines[3].move_copy_to(lines[6], src_letter_indices=[1, 2, 3, 4, 5, 6, 7, 8], new_color='text',
                              target_letter_indices=[1, 2, 3, 4, 5, 6, 7, 8], begin_time=t0, transition_time=dt)
        t0 += sep
        lines[5].move_copy_to(lines[6], src_letter_indices=[10, 11], target_letter_indices=[9, 10], begin_time=t0,
                              transition_time=dt, new_color='text')
        lines[6].write(letter_set=[10, 11], begin_time=t0 + 3 * dt / 4, transition_time=1)
        lines[5].copies_of_letters[-1].disappear(begin_time=t0 + dt, transition_time=dt_dis)
        t0 += sep
        lines[4].write(letter_range=[10, 19], begin_time=t0, transition_time=0)
        lines[4].move_copy_to(lines[6], src_letter_indices=[10, 11, 12, 13, 14, 15, 16, 17, 18],
                              target_letter_indices=[12, 13, 14, 15, 16, 17, 18, 19, 20], begin_time=t0,
                              transition_time=dt, new_color='text')

        t0 += sep + 1
        s_m4.disappear(begin_time=t0)
        s_c_dash.disappear(begin_time=t0 + 1, transition_time=0)
        s_c_dash2 = Sphere(0.25, location=cp_loc, name='C^\prime', color='joker')
        s_c_dash2.write_name_as_label(modus='down_left', offset=[-0.17, 0, 0.28], begin_time=t0, transition_time=0)

        s_c_dash2.grow(begin_time=t0, transition_time=0)

        transition_line = SimpleTexBObject(r"\text{Line }BC^\prime:", color='joker', name='Line_Trans')
        display.add_text_in(transition_line, line=7, indent=0.5)

        t0 += 1.5

        transition_line.write(begin_time=t0)
        t0 += 1.5

        line_bcp = Cylinder.from_start_to_end(start=b_loc, end=cp_loc, color='joker', thickness=0.5, name='LCP')
        line_bcp.grow(modus='from_start', begin_time=t0)

        t0 += 1
        duration = 3
        v = cp_loc - b_loc

        s_c_dash2.move_to(target_location=cp_loc + v, begin_time=t0)
        s_c_dash2.move_to(target_location=cp_loc - 0.5 * v, begin_time=t0 + 1, transition_time=1.5)
        s_c_dash2.move_to(target_location=cp_loc, begin_time=t0 + 2.5, transition_time=0.5)

        n = 60
        steps = int(duration * FRAME_RATE / n)
        dt = duration / n

        coords.add_objects([s_m, s_m2, s_m4, l_ac, line_bcp, bac, rotation, rot_box2, s_c_dash2])

        for frame in range(int(t0 * FRAME_RATE), int((t0 + duration) * FRAME_RATE) + 1, steps):
            location = ibpy.get_location_at_frame(s_c_dash2, frame)
            line_bcp.move_end_point(target_location=location, begin_time=frame / FRAME_RATE - dt, transition_time=dt)

        t0 += duration + 0.5
        line_bcp.disappear(begin_time=t0)

        t0 += 1.5

        print("last frame", t0 * FRAME_RATE)

    def solution1c(self):
        cues = self.sub_scenes['solution1c']
        t0 = 0  # cues['start']

        debug = False
        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # coord
        details = 3
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 1], [-1, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[3, 3],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2., 1.1, 1), np.arange(-1, 2.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[2, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)

        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18)
        title0 = SimpleTexBObject(r"\text{Solution -- Part 1}", color='important', aligned='center')
        display.set_title(title0)
        title0.write(begin_time=t0, transition_time=0)

        colors1 = flatten([['joker'] * 2, ['text'] * 10])

        lines = [
            SimpleTexBObject(r"\text{Rotation of }C:", aligned='left', color='joker', name='Line0'),
            SimpleTexBObject(r"C-B=x-1+yi", color='text', name='Line1'),
            SimpleTexBObject(r"(C-B)\cdot C=((x-1)+yi)\cdot(x+yi)", color='text', name='Line2'),
            SimpleTexBObject(r"=(x^2-x-y^2)+(2xy-y)i", color='text', name='Line3'),  # dummy line for replacement
            SimpleTexBObject(r"C^\prime=(C-B)\cdot C+B", color=colors1, name='Line4'),
            SimpleTexBObject(r"=(x^2-x-y^2+1)+(2xy-y)i", color='text', name='Line5'),
            SimpleTexBObject(r"\text{Line }BC^\prime:", color='joker', name='Line6'),
            SimpleTexBObject(r"C'-B=(x^2-x-y^2)+(2xy-y)i", color='text', name='Line7'),
            SimpleTexBObject(r"\lambda(C'-B)=(\lambda x^2-\lambda x-\lambda y^2)+(2\lambda xy-\lambda y)i",
                             color='text', name='Line8'),
            SimpleTexBObject(r"BC^\prime: (\lambda x^2-\lambda x-\lambda y^2+1)+(2\lambda xy-\lambda y)i",
                             color='joker', name='Line9'),
            SimpleTexBObject(r"\text{Line }AC:", color='example', name='Line10'),
            SimpleTexBObject(r"AC: \mu x+\mu y i", color='example', name='Line11'),
        ]
        indents = [0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 1.4, 0.5, 0.5, 0.5]
        rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]

        count = 0
        for line, row, indent in zip(lines, rows, indents):
            if count == 8:
                scale = 0.5
            else:
                scale = 0.7
            display.add_text_in(line, line=row, indent=indent, scale=scale)
            count += 1

        lines[2].align(lines[1], char_index=7, other_char_index=3)
        lines[3].align(lines[1], char_index=0, other_char_index=3)
        lines[4].align(lines[1], char_index=2, other_char_index=3)
        lines[5].align(lines[1], char_index=0, other_char_index=3)
        lines[7].align(lines[1], char_index=4, other_char_index=3)

        for i in range(7):
            lines[i].write(begin_time=t0, transition_time=0)

        # objects in coordinate system
        a_loc = coords.coords2location([0, 0])
        s_a = Sphere(0.25, location=a_loc, name='A=0')
        s_a.grow(begin_time=t0, transition_time=0)
        s_a.write_name_as_label(begin_time=t0, modus='down_left', transition_time=0)

        b_loc = coords.coords2location([1, 0])
        s_b = Sphere(0.25, location=b_loc, name='B=1')
        s_b.grow(begin_time=t0, transition_time=0)
        s_b.write_name_as_label(begin_time=t0, modus='down_right', transition_time=0)

        l_ab = Cylinder.from_start_to_end(start=a_loc, end=b_loc, thickness=0.5)
        l_ab.grow(begin_time=t0, modus='from_start', transition_time=0)

        c_loc = coords.coords2location([0.75, 1.5])
        s_c = Sphere(0.25, location=c_loc, name='C=x+y\,i', color='joker')
        s_c.grow(begin_time=t0, transition_time=0)
        s_c.write_name_as_label(begin_time=t0, transition_time=0, modus='down_left', offset=[-0.59, 0, 0])

        # continue construction
        zc = 0.75 + 1.5j
        zb = 1
        zcp = zc * (zc - zb) / np.abs(zc) + zb
        cp_loc = coords.coords2location([np.real(zcp), np.imag(zcp)])
        s_c_dash = Sphere(0.25, location=cp_loc, name='C^\prime', color='joker')
        s_c_dash.write_name_as_label(modus='down_left', offset=[-0.17, 0, 0.28], begin_time=t0, transition_time=0)
        t0 += 1

        # shift by B

        arrows = [
            PArrow(start=b_loc, end=Vector(), name='Arrow1', color='example', thickness=2),
            PArrow(start=b_loc, end=Vector(), name='Arrow2', color='example', thickness=2),
        ]

        coords.add_objects(arrows)

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=1)

        t0 += 1.5

        end_positions = [cp_loc - b_loc, Vector()]

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]
        coords.add_objects(containers)

        for container in containers:
            container.appear(begin_time=t0)

        s_m = Sphere(0.2, location=cp_loc, color='text', name=r"C^\prime-B")
        s_m2 = Sphere(0.2, location=b_loc, color='text', name='B-B')

        spheres = [s_m2, s_m]
        modi = ['down_right', 'down_left']
        for modus, sphere in zip(modi, spheres):
            sphere.move(direction=-b_loc, begin_time=t0)
            sphere.write_name_as_label(modus=modus, begin_time=t0)

        for container in containers:
            container.shrink(begin_time=t0)

        t0 += 1.5
        lines[7].write(letter_range=[0, 5], begin_time=t0, transition_time=0.5)
        t0 += 0.6

        lines[5].move_copy_to(lines[7], src_letter_indices=[1, 2, 3, 4, 5, 6, 7, 8, 11],
                              target_letter_indices=[5, 6, 7, 8, 9, 10, 11, 12, 13], begin_time=t0,
                              transition_time=1, new_color='text')

        t0 += 1.5
        lines[5].move_copy_to(lines[7], src_letter_indices=[12, 13, 14, 15, 16, 17, 18, 19, 20],
                              target_letter_indices=[14, 15, 16, 17, 18, 19, 20, 21, 22], begin_time=t0,
                              transition_time=1, new_color='text')

        t0 += 1.5

        # draw lambda line
        duration = 5

        v = cp_loc - b_loc
        tracer = Sphere(0.1, location=-v, color='joker')
        tracer.grow(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        line = Cylinder.from_start_to_end(start=-v, end=2.1 * v, color='text', thickness=0.5)
        line.grow(modus='from_start', begin_time=t0, transition_time=duration)
        tracer.move(direction=3.1 * v, begin_time=t0, transition_time=duration)

        l_old = np.Infinity
        labels = []
        movers = [line]
        removables = []
        count = 0
        for i in range(-2, 6):
            labels.append(r"\lambda=" + str(i / 2))
        for frame in range(int(t0 * FRAME_RATE), int((t0 + duration) * FRAME_RATE) + 1):
            location = ibpy.get_location_at_frame(tracer, frame)
            l = np.floor(-2 * np.sign(location.x) * location.length / v.length) / 2
            if l_old != l:
                l_old = l
                sphere = Sphere(0.15, location=location, name=labels[count], color='text')
                t = frame / FRAME_RATE
                sphere.grow(begin_time=t, transition_time=0.1)
                pos = 'up_right'
                # if count == 0:
                #     pos = 'up'
                # else:
                #     pos = 'up_left'
                count += 1
                sphere.write_name_as_label(modus=pos, begin_time=t + 0.1, transition_time=0.2)
                movers.append(sphere)
                removables.append(sphere)
                coords.add_object(sphere)

        coords.add_objects([s_a, s_b, l_ab, s_c, s_c_dash, s_m2, s_m, line, tracer])

        t0 += duration + 1

        lines[8].write(begin_time=t0, transition_time=2)
        t0 += 2

        # shift back

        arrows = [
            PArrow(start=-b_loc, end=Vector(), name='Arrow3', color='example', thickness=2),
            PArrow(start=-b_loc, end=Vector(), name='Arrow4', color='example', thickness=2),
        ]

        end_positions = [b_loc, cp_loc]

        for i in range(1, len(movers)):
            arrows.append(
                PArrow(start=-b_loc, end=Vector(), name='Arrow' + str(i + 4), color='example', thickness=2),
            )
            end_positions.append((i / 2 - 1.5) * v + b_loc)

        coords.add_objects(arrows)

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=1)

        t0 += 1.5

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]
        coords.add_objects(containers)

        for container in containers:
            container.appear(begin_time=t0)

        for sphere in spheres:
            sphere.move(direction=b_loc, begin_time=t0)
            sphere.label_disappear(begin_time=t0)

        for mover in movers:
            mover.move(direction=b_loc, begin_time=t0)
            mover.change_color(new_color='joker', begin_time=t0)
            if mover.label:
                mover.label.change_color(new_color='joker', begin_time=t0)

        for container in containers:
            container.shrink(begin_time=t0)

        t0 += 1.5

        lines[9].write(begin_time=t0, transition_time=2)
        t0 += 2.5

        lines[10].write(begin_time=t0, transition_time=1)
        display.move(direction=[0, -1, 0], begin_time=t0)
        t0 += 1.5

        # draw second line

        v = c_loc
        tracer2 = Sphere(0.1, location=-0.9 * v, color='example')
        tracer2.grow(begin_time=t0, transition_time=0.1)
        t0 += 0.5

        line2 = Cylinder.from_start_to_end(start=-0.9 * v, end=1.6 * v, color='example', thickness=0.5)
        line2.grow(modus='from_start', begin_time=t0, transition_time=duration)
        tracer2.move(direction=2.5 * v, begin_time=t0, transition_time=duration)
        removables.append(tracer2)

        coords.add_objects([line2, tracer2])
        m_old = np.Infinity
        labels = []
        count = 0
        for i in range(-2, 5):
            labels.append(r"\mu=" + str(i / 2))

        for frame in range(int(t0 * FRAME_RATE), int((t0 + duration) * FRAME_RATE) + 1):
            location = ibpy.get_location_at_frame(tracer2, frame)
            m = np.floor(2 * np.sign(location.x) * location.length / v.length) / 2
            if m_old != m:
                m_old = m
                sphere = Sphere(0.15, location=location, name=labels[count], color='example')
                t = frame / FRAME_RATE
                sphere.grow(begin_time=t, transition_time=0.1)
                pos = 'up_left'
                count += 1
                sphere.write_name_as_label(modus=pos, begin_time=t + 0.1, transition_time=0.2, aligned='right')
                removables.append(sphere)
                coords.add_object(sphere)

        t0 += duration + 1

        lines[11].write(begin_time=t0)
        t0 += 1.5

        duration = 5
        dt = 2 * duration / len(removables)
        for rem in removables:
            rem.disappear(begin_time=t0, transition_time=dt)
            t0 += dt / 2

        t0 += 1

        title = SimpleTexBObject(r"\text{Solution -- Part 1}", color='important', aligned='center')
        display.set_title_back(title)
        title.write(begin_time=t0, transition_time=0)

        colors1 = flatten([['joker'] * 2, ['text'] * 10])

        lines = [
            SimpleTexBObject(r"BC^\prime: (\lambda x^2-\lambda x-\lambda y^2+1)+(2\lambda xy-\lambda y)i",
                             color='joker', name='Line1'),
            SimpleTexBObject(r"AC: \mu x+\mu y i", color='example', name='Line2'),
        ]
        indents = [0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 1.4, 0.5, 0.5, 0.5]
        rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]

        for line, row, indent in zip(lines, rows, indents):
            display.add_text_in_back(line, line=row, indent=indent, scale=scale)

        for line in lines:
            line.write(begin_time=t0, transition_time=0)

        display.rotate(rotation_euler=[np.pi / 2, 0, np.pi - np.pi / 15], begin_time=t0)
        t0 += 2

        print("last frame ", t0 * FRAME_RATE)

    def solution1d(self):
        cues = self.sub_scenes['solution1d']
        t0 = 0  # cues['start']

        debug = False
        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # coord
        details = 3
        coords = CoordinateSystem(dim=2, lengths=[10, 10], domains=[[-2, 1], [-1, 2]], radii=[0.03, 0.03],
                                  all_n_tics=[3, 3],
                                  labels=[r"\mathbb{R}", r"i\cdot\mathbb{R}"],
                                  all_tic_labels=[np.arange(-2., 1.1, 1), np.arange(-1, 2.1, 1)],
                                  colors=['text', 'text'], label_colors=['text', 'text'],
                                  label_digits=[0, 0], label_units=['', 'i'],
                                  axis_label_size='medium',
                                  tic_label_size='small',
                                  location_of_origin=[2, 0, -2],
                                  name='ComplexPlane')
        coords.appear(begin_time=t0, transition_time=0)

        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18)
        display.move(direction=[0, -1, 0], begin_time=t0, transition_time=0)
        title0 = SimpleTexBObject(r"\text{Solution -- Part 1}", color='important', aligned='center')
        display.set_title(title0)
        title0.write(begin_time=t0, transition_time=0)

        # objects in coordinate system
        a_loc = coords.coords2location([0, 0])
        s_a = Sphere(0.25, location=a_loc, name='A=0')
        s_a.grow(begin_time=t0, transition_time=0)
        s_a.write_name_as_label(begin_time=t0, modus='down_left', transition_time=0)

        b_loc = coords.coords2location([1, 0])
        s_b = Sphere(0.25, location=b_loc, name='B=1')
        s_b.grow(begin_time=t0, transition_time=0)
        s_b.write_name_as_label(begin_time=t0, modus='down_right', transition_time=0)

        l_ab = Cylinder.from_start_to_end(start=a_loc, end=b_loc, thickness=0.5)
        l_ab.grow(begin_time=t0, modus='from_start', transition_time=0)

        c_loc = coords.coords2location([0.75, 1.5])
        s_c = Sphere(0.25, location=c_loc, name='C=x+y\,i', color='joker')
        s_c.grow(begin_time=t0, transition_time=0)
        s_c.write_name_as_label(begin_time=t0, transition_time=0, modus='down_left', offset=[-0.59, 0, 0])

        # continue construction
        zc = 0.75 + 1.5j
        zb = 1
        zcp = zc * (zc - zb) / np.abs(zc) + zb
        cp_loc = coords.coords2location([np.real(zcp), np.imag(zcp)])
        s_c_dash = Sphere(0.25, location=cp_loc, name='C^\prime', color='joker')
        s_c_dash.write_name_as_label(modus='down_left', offset=[-0.17, 0, 0.28], begin_time=t0, transition_time=0)

        # draw lambda line

        v = cp_loc - b_loc

        l_bcp = Cylinder.from_start_to_end(start=-v + b_loc, end=2.1 * v + b_loc, color='joker', thickness=0.5)
        l_bcp.grow(modus='from_start', begin_time=t0, transition_time=0)

        # draw second line
        v = c_loc
        l_ac = Cylinder.from_start_to_end(start=-0.9 * v, end=1.6 * v, color='example', thickness=0.5)
        l_ac.grow(modus='from_start', begin_time=t0, transition_time=0)

        if not debug:
            lines = [
                SimpleTexBObject(r"BC^\prime: (\lambda x^2-\lambda x-\lambda y^2+1)+(2\lambda xy-\lambda y)i",
                                 color='joker', name='Line0'),
                SimpleTexBObject(r"AC: \mu x+\mu y i", color='example', name='Line1'),
                SimpleTexBObject(r"\mu x =\lambda x^2-\lambda x-\lambda y^2+1", color='text', name='Line2'),
                SimpleTexBObject(r"\mu y =2\lambda x y-\lambda y", color='text', name='Line3'),
                SimpleTexBObject(r"\mu =2\lambda x-\lambda", color='text', name='Line3b'),
                SimpleTexBObject(r"(2\lambda x-\lambda)x=\lambda x^2-\lambda x-\lambda y^2+1", color='text',
                                 name='Line5'),
                SimpleTexBObject(r"2\lambda x^2-\lambda x=\lambda x^2-\lambda x-\lambda y^2+1", color='text',
                                 name='Line5b'),
                SimpleTexBObject(r"1\lambda x^2+\lambda y^2=1", color='text', name='Line7'),
                SimpleTexBObject(r"\lambda( x^2+y^2)=1", color='text', name='Line7b'),
                SimpleTexBObject(r"\lambda={1\over x^2+y^2}", color='text', name='Line9'),
                SimpleTexBObject(r"\mu =2\lambda x-\lambda", color='text', name='Line10'),
                SimpleTexBObject(r"\mu =\lambda(2 x-1)", color='text', name='Line10b'),
                SimpleTexBObject(r"\mu ={2 x-1\over x^2+y^2}", color='text', name='Line12'),
            ]
            indents = [0.5, 0.5, 2.5, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            rows = [1, 2, 3, 4, 4, 6, 6, 7.5, 7.5, 7.5, 10, 10, 13]

            for line, row, indent in zip(lines, rows, indents):
                display.add_text_in(line, line=row, indent=indent)

            lines[4].align(lines[3], char_index=1, other_char_index=2)
            lines[5].align(lines[3], char_index=8, other_char_index=2)
            lines[6].align(lines[3], char_index=7, other_char_index=2)
            lines[7].align(lines[3], char_index=8, other_char_index=2)
            lines[8].align(lines[3], char_index=8, other_char_index=2)
            lines[9].align(lines[3], char_index=1, other_char_index=2)
            lines[10].align(lines[3], char_index=1, other_char_index=2)
            lines[11].align(lines[3], char_index=1, other_char_index=2)
            lines[12].align(lines[3], char_index=1, other_char_index=2)

            for i in range(2):
                lines[i].write(begin_time=t0, transition_time=0)

            t0 += 1

        # rotate coordinate system

        coords.rotate(rotation_euler=[-np.pi / 3, 0, 0], begin_time=t0, transition_time=2)
        coords.move(direction=[0, 0, 0], begin_time=t0, transition_time=2)
        t0 += 2.5

        # position empty flags

        t_loc = Vector(coords.coords2location([0.75, 0]))
        result = geometry(a_loc, b_loc, c_loc, t_loc)
        d_loc = result[0]

        v1 = cp_loc - b_loc
        v2 = c_loc

        duration = 70
        if not debug:
            flags = [
                Flag(colors=['important', 'joker'], name='FlagLambda', rotation_euler=[np.pi / 3, 0, 0],
                     location=d_loc + v1, simulation_start=0,
                     simulation_duration=duration),
                Flag(colors=['important', 'example'], name='FlagMu', mirror=True, rotation_euler=[-np.pi / 3, 0, np.pi],
                     location=d_loc - v2,
                     simulation_start=0,
                     simulation_duration=duration,
                     scale=0.975),
                Flag(colors=['important', 'example'], name='FlagMu0', mirror=True,
                     rotation_euler=[-np.pi / 3, 0, np.pi],
                     location=a_loc - v2,
                     simulation_start=0,
                     simulation_duration=duration, scale=0.5),
                Flag(colors=['important', 'example'], name='FlagMu1', mirror=True,
                     rotation_euler=[-np.pi / 3, 0, np.pi],
                     location=c_loc + v2,
                     simulation_start=0,
                     simulation_duration=duration, scale=0.5),
            ]
            flags[0].appear(begin_time=t0, transition_time=0.5)
            flags[1].appear(begin_time=t0, transition_time=0.5)
            flags[0].move(direction=-v1, begin_time=t0, transition_time=2)
            flags[1].move(direction=v2, begin_time=t0, transition_time=2)

        s_d = Sphere(0.25, location=d_loc, color='important', name='D')
        s_d.grow(begin_time=t0, transition_time=0.5)
        s_d.write_name_as_label(begin_time=t0, modus='up_right')

        t0 += 2.5

        if not debug:
            # lse
            lines[1].move_copy_to(lines[2], src_letter_indices=[3, 4], target_letter_indices=[0, 1], new_color='text',
                                  begin_time=t0)
            t0 += 1
            lines[2].write(letter_set=[2], begin_time=t0, transition_time=0.1)
            t0 += 0.2
            lines[0].move_copy_to(lines[2], src_letter_indices=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                  target_letter_indices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], new_color='text',
                                  begin_time=t0)
            t0 += 1.5

            lines[1].move_copy_to(lines[3], src_letter_indices=[6, 7], target_letter_indices=[0, 1], new_color='text',
                                  begin_time=t0)
            for l in lines[1].copies_of_letters:
                l.disappear(begin_time=t0 + 2.7, transition_time=0)
            t0 += 1
            lines[3].write(letter_set=[2], begin_time=t0, transition_time=0.1)
            t0 += 0.2
            lines[0].move_copy_to(lines[3], src_letter_indices=[20, 21, 22, 23, 24, 25, 26],
                                  target_letter_indices=[3, 4, 5, 6, 7, 8, 9], new_color='text',
                                  begin_time=t0)
            for l in lines[0].copies_of_letters:
                l.disappear(begin_time=t0 + 1.5, transition_time=0)
            t0 += 1.5

            lines[3].write(letter_set=[0, 1, 3, 4, 5, 6, 7, 8, 9], begin_time=t0, transition_time=0)
            lines[3].replace(lines[4], begin_time=t0)
            t0 += 1.5

            underline = Cylinder(length=1.1, location=[0, 0.3, 0.025], thickness=0.1, rotation_euler=[0, np.pi / 2, 0],
                                 color='text', name="underline")
            display.add_child(underline)

            underline.grow(begin_time=t0)

            t0 += 1.5

            # calculate lambda
            lines[4].write(begin_time=t0, transition_time=0)
            lines[3].move(direction=[0, 0, -0.025], begin_time=t0)  # make it invisible
            lines[4].letters[0].change_color(new_color='important', begin_time=t0)
            lines[2].write(letter_set=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], begin_time=t0, transition_time=0)
            lines[2].letters[0].change_color(new_color='important', begin_time=t0)

            t0 += 1.5

            lines[4].move_copy_to(lines[5], src_letter_indices=[2, 3, 4, 5, 6], target_letter_indices=[1, 2, 3, 4, 5],
                                  begin_time=t0)
            for l in lines[4].copies_of_letters:
                l.disappear(begin_time=t0 + 4.5)
            t0 += 1
            # ()
            lines[5].write(letter_set=[0, 6], begin_time=t0, transition_time=0.1)
            t0 += 0.5
            lines[2].move_copy_to(lines[5], src_letter_indices=[1],
                                  target_letter_indices=[7], begin_time=t0)
            for l in lines[2].copies_of_letters:
                l.disappear(begin_time=t0 + 3)
            t0 += 1
            # =
            lines[5].write(letter_set=[8], begin_time=t0, transition_time=0.1)
            t0 += 0.5
            lines[2].move_copy_to(lines[5], src_letter_indices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                  target_letter_indices=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], begin_time=t0)
            for l in lines[2].copies_of_letters:
                l.disappear(begin_time=t0 + 1.5)
            t0 += 1.5

            lines[5].write(letter_set=[1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], begin_time=t0,
                           transition_time=0)
            lines[5].replace(lines[6], src_letter_range=[0, 8], img_letter_range=[0, 7], begin_time=t0)

            t0 += 1.5
            lines[6].write(begin_time=t0, transition_time=0)
            lines[5].move(direction=[0, 0, -0.025], begin_time=t0)  # hide in display

            # remove -lambda x
            set = [4, 5, 6, 11, 12, 13]
            for s in set:
                lines[6].letters[s].change_color(new_color='important', begin_time=t0)
            t0 += 1.5
            for s in set:
                lines[6].letters[s].disappear(begin_time=t0)

            t0 += 1.5

            # terms with x**2
            lines[6].move_copy_to_and_remove(lines[7], src_letter_indices=[0], target_letter_indices=[0],
                                             new_color='example', begin_time=t0, remove_time=t0)
            lines[6].move_copy_to_and_remove(lines[7], src_letter_indices=[1, 2, 3], target_letter_indices=[1, 2, 3],
                                             new_color='example', begin_time=t0)
            lines[6].move_copy_to_and_remove(lines[7], src_letter_indices=[8, 9, 10], target_letter_indices=[1, 2, 3],
                                             new_color='example', begin_time=t0)
            t0 += 1
            lines[7].write(letter_set=[1, 2, 3], begin_time=t0, transition_time=0)
            t0 += 1
            # convert minus into plus
            lines[6].move_copy_to_and_remove(lines[7], src_letter_indices=[14, 15, 16, 17],
                                             target_letter_indices=[4, 5, 6, 7], begin_time=t0,
                                             new_color='example')
            t0 += 1
            lines[7].write(letter_set=[4, 5, 6, 7], begin_time=t0, transition_time=0)
            t0 += 1
            lines[7].write(letter_set=[8], begin_time=t0, transition_time=0.1)
            t0 += 0.5
            lines[6].move_copy_to_and_remove(lines[7], src_letter_indices=[19], target_letter_indices=[9],
                                             begin_time=t0, new_color='example')
            t0 += 1
            lines[7].write(letter_set=[9], begin_time=t0, transition_time=0.1)
            t0 += 0.5

            # there is a place holder at the position 0 therefore src=[1,8]
            lines[7].replace(lines[8], src_letter_range=[1, 8], img_letter_range=[0, 8], begin_time=t0)
            t0 += 1.5
            lines[8].write(begin_time=t0, transition_time=0)
            lines[7].move(direction=[0, 0, -0.025], begin_time=t0)
            t0 += 1.5

            lines[8].replace(lines[9], begin_time=t0)
            t0 += 1.5

            flags[0].set_text(r"\fbox{$\phantom{\lambda={1\over x^2+y^2}}$}", begin_time=0)
            flags[0].set_text(r"\fbox{$\lambda={1\over x^2+y^2}$}",
                              begin_time=t0 + 0.1)  # set text twice because the first image is not  mixed

            t0 += 1.5

            lines[10].write(begin_time=t0)
            t0 += 1.5
            lines[10].replace(lines[11], begin_time=t0)
            t0 += 1.5

            # highlight lambdas
            lines[9].write(begin_time=t0, transition_time=0)
            lines[8].move(direction=[0, 0, -0.025], begin_time=t0)
            lines[11].write(begin_time=t0, transition_time=0)
            lines[10].move(direction=[0, 0, -0.025], begin_time=t0)
            lines[11].letters[2].change_color(new_color='important', begin_time=t0)
            lines[9].letters[0].change_color(new_color='important', begin_time=t0)
            t0 += 1.5

            lines[12].write(letter_set=[0, 1], begin_time=t0, transition_time=0.2)
            t0 += 0.5
            lines[11].move_copy_to(lines[12], src_letter_indices=[4, 5, 6, 7], target_letter_indices=[2, 5, 8, 10],
                                   begin_time=t0, offset=[0, 0.24, 0])
            t0 += 1.5
            lines[9].move_copy_to(lines[12], src_letter_indices=[5, 2, 3, 6, 7, 8],
                                  target_letter_indices=[6, 3, 4, 7, 9, 11],
                                  begin_time=t0)  # first letter is the fraction bar
            t0 += 1.5

            flags[1].set_text(r"\fbox{$\phantom{\mu={2x-1\over x^2+y^2}}$}", begin_time=0)
            flags[1].set_text(r"\fbox{$\mu={2x-1\over x^2+y^2}$}",
                              begin_time=t0 + 0.1)  # set text twice because the first image is not  mixed
            t0 += 1.5

            flags[2].appear(begin_time=t0, transition_time=0.5)
            flags[3].appear(begin_time=t0, transition_time=0.5)
            flags[2].move(direction=v2, begin_time=t0, transition_time=2)
            flags[3].move(direction=-v2, begin_time=t0, transition_time=2)
            t0 += 2
            flags[2].set_text(r"\fbox{$\phantom{\mu=0}$}", begin_time=0)
            flags[2].set_text(r"\fbox{$\mu=0$}", begin_time=t0)
            flags[3].set_text(r"\fbox{$\phantom{\mu=1}$}", begin_time=0)
            flags[3].set_text(r"\fbox{$\mu=1$}", begin_time=t0)
            t0 += 1.5

            # add objects to coordinate system
            coords.add_objects(flags)

        back_title = SimpleTexBObject(r"\text{Solution -- Part 1}", color="important", aligned='center')
        display.set_title_back(back_title)
        back_title.write(begin_time=t0, transition_time=0)
        if not debug:
            flags[0].disappear(begin_time=t0)

        back_lines = [
            SimpleTexBObject(r"\mu ={2 x-1\over x^2+y^2}", color='text', name='BLine0'),
            SimpleTexBObject(r"x>0.5\Longrightarrow 2x-1>0 \Longrightarrow \mu>0", color='text', name='BLine1'),
            SimpleTexBObject(r"(x-1)^2\ge 0", color='text', name='BLine2'),
            SimpleTexBObject(r"x^2-2x+1\ge 0", color='text', name='BLine2b'),
            SimpleTexBObject(r"x^2\ge 2x-1", color='text', name='BLine4'),
            SimpleTexBObject(r"1\ge {2x-1\over x^2}", color='text', name='BLine5'),
            SimpleTexBObject(r"1> {2x-1\over x^2+y^2}\Longrightarrow \mu<1", color='text', name='BLine6'),
            SimpleTexBObject(r"\text{\textcircled{$\checkmark$}}", color='important', name='Check'),
        ]
        indents = [2.5, 0.5, 1.4, 2.5, 0, 0, 0, 8]
        rows = [1, 3, 5, 5, 5, 7, 9, 12]
        scales = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1.4]

        for line, row, indent, scale in zip(back_lines, rows, indents, scales):
            display.add_text_in_back(line, line=row, indent=indent, scale=scale)

        back_lines[3].align(back_lines[2], char_index=7, other_char_index=6)
        back_lines[4].align(back_lines[2], char_index=2, other_char_index=6)
        back_lines[5].align(back_lines[2], char_index=1, other_char_index=6)
        back_lines[6].align(back_lines[2], char_index=1, other_char_index=6)

        back_lines[0].write(begin_time=t0, transition_time=0)

        display.turn(begin_time=t0)
        t0 += 1.5

        back_lines[1].write(letter_range=[0, 13], begin_time=t0, transition_time=1)
        t0 += 1.25
        back_lines[1].write(letter_range=[13, 18], begin_time=t0, transition_time=1)
        t0 += 0.5
        for s in [15, 16, 17]:
            back_lines[1].letters[s].change_color(new_color="example", begin_time=t0, transition_time=0.5)
        t0 += 0.75

        back_lines[2].write(letter_range=[0, 6], begin_time=t0, transition_time=0.75)
        t0 += 1
        back_lines[2].write(letter_range=[6, 8], begin_time=t0, transition_time=0.25)
        t0 += 1.5
        back_lines[2].replace(back_lines[3], begin_time=t0)
        t0 += 1.5
        back_lines[3].write(begin_time=t0, transition_time=0)
        back_lines[2].move(direction=[0, 0, 0.025], begin_time=t0)
        t0 += 1.5
        back_lines[3].replace(back_lines[4], begin_time=t0)
        t0 += 1.5
        back_lines[4].write(begin_time=t0, transition_time=0)
        back_lines[3].move(direction=[0, 0, 0.025], begin_time=t0)
        t0 += 1.5
        back_lines[5].write(letter_set=[0, 1], begin_time=t0, transition_time=0.2)
        t0 += 0.3
        back_lines[4].move_copy_to(back_lines[5], src_letter_indices=[3, 4, 5, 6]
                                   , target_letter_indices=[2, 3, 6, 8], offset=[0, 0.25, 0],
                                   begin_time=t0)
        t0 += 1.5
        back_lines[5].write(letter_set=[4], begin_time=t0, transition_time=0.3)
        t0 += 0.5
        back_lines[4].move_copy_to(back_lines[5], src_letter_indices=[0, 1]
                                   , target_letter_indices=[5, 7], offset=[0, -0.25, 0],
                                   begin_time=t0)
        t0 += 1.5
        back_lines[6].write(letter_range=[0, 12], begin_time=t0, transition_time=1)
        t0 += 1.25
        back_lines[6].write(letter_range=[12, 17], begin_time=t0, transition_time=0.5)
        for s in [14, 15, 16]:
            back_lines[6].letters[s].change_color(new_color="example", begin_time=t0, transition_time=0.5)
        t0 += 0.75

        back_lines[7].write(begin_time=t0, transition_time=0.5)
        t0 += 1

        coords.add_objects([l_ab, l_ac, l_bcp, s_c_dash, s_c, s_a, s_b, s_d])

        print("last frame ", t0 * FRAME_RATE)

    def solution2(self):
        cues = self.sub_scenes['solution2']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # Repeat construction of the question
        title = SimpleTexBObject(r'\text{Solution -- Part 2}', color='important', aligned='center')
        [a, b, c, t] = [0, 1, 0.75 + 1.5j, 0.75]

        # the construction is outsourced
        d_title = 2
        d_problem = 2
        d_part1 = 2
        d_part2 = 4
        dic = self.construction(t0, title, locs=[a, b, c, t], parts=[True,True,True],durations=[d_title, d_problem, d_part1, d_part2])

        # retrieve variables
        coords = dic['coords']
        lines = dic['lines']
        sphere_strings = ['s_a', 's_b', 's_c', 's_d', 's_e', 's_t']
        segment_strings = ['l_ab', 'l_ac', 'l_bc', 'l_bd', 'l_de', 'l_tc']
        arc_strings = ['cbd', 'bac','ra','ra2']
        displays = ['display0', 'display1']

        spheres = [s_a, s_b, s_c, s_d, s_e, s_t] = get_from_dictionary(dic, sphere_strings)
        segments = [l_ab, l_ac, l_bc, l_bd, l_de, l_tc] = get_from_dictionary(dic, segment_strings)
        [cbd, bac,ra,ra2] = get_from_dictionary(dic, arc_strings)
        displays = get_from_dictionary(dic, displays)

        a_loc = coords.coords2location(z2p(a))
        b_loc = coords.coords2location(z2p(b))
        t_loc = coords.coords2location(z2p(t))
        c_loc = coords.coords2location(z2p(c))

        # make old lines fade
        t0 += d_title + d_problem
        s_d.write_name_as_label(modus='up', name='D', begin_time=t0 + d_part1 / 4, transition_time=0.2)
        t0 += d_part1
        for i in range(0, 10):
            lines[i].disappear(begin_time=t0 + 1, alpha=0.125)

        t0 += d_part2 / 2
        s_e.write_name_as_label(modus='down', name='E', begin_time=t0 + d_part2 / 4, transition_time=0.2)
        t0 += d_part2 / 2 + 0.5

        # start dynamics
        d_anim = 5
        d_zoom = 2

        direction = to_vector([0, 0, -3])
        s_c.move(direction=direction, begin_time=t0, transition_time=d_anim)

        steps = 10
        dt = (d_anim) / steps
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = geometry(a_loc, b_loc, c_new, t_loc)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bac.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            l_bc.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            l_bd.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            ra2.rotate(rotation_euler=[0, np.pi/2 - alpha, 0], begin_time=t, transition_time=dt)
            ra2.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += d_anim

        # zoom in
        coords.zoom(zoom=2, begin_time=t0, transition_time=d_zoom)
        coords.move(direction=[-4, 0, -2], begin_time=t0, transition_time=d_zoom)
        for sphere in spheres:
            sphere.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0, transition_time=d_zoom)
        for seg in segments:
            seg.rescale(rescale=[0.5, 0.5, 1], begin_time=t0, transition_time=d_zoom)
        t0 += d_zoom

        s_c.move(direction=-direction, begin_time=t0, transition_time=d_anim)
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = geometry(a_loc, b_loc, c_new, t_loc)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bac.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            l_bc.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            l_bd.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            ra2.rotate(rotation_euler=[0, np.pi/2 - alpha, 0], begin_time=t, transition_time=dt)
            ra2.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += d_anim

        # prepare transition
        s_a.write_name_as_label(modus='down_right', name='A=0', begin_time=t0, transition_time=0.6)
        s_b.write_name_as_label(modus='down_right', name='B=1', begin_time=t0 + 0.3, transition_time=0.6)
        s_c.write_name_as_label(modus='up', name='C=x+yi', begin_time=t0 + 0.6, transition_time=0.6)

        l_tc.disappear(begin_time=t0 + 1)
        l_bd.disappear(begin_time=t0 + 1)
        l_bc.disappear(begin_time=t0 + 1)
        s_t.disappear(begin_time=t0 + 1)
        bac.disappear(begin_time=t0 + 1)
        cbd.disappear(begin_time=t0 + 1)
        ra.disappear(begin_time=t0+1)
        ra2.disappear(begin_time=t0+1)

        t0 += 2.5

        title_back = SimpleTexBObject(r"\text{Solution -- Part 2}", color='important', aligned='center')
        displays[0].set_title_back(title_back)
        title_back.write(begin_time=0, transition_time=0)

        for display in displays:
            display.turn(begin_time=t0)

        t0 += 1.5

        line = SimpleTexBObject(r"\text{Coordinates of }D")
        displays[0].add_text_in_back(line, line=1, indent=0.5)
        line.write(begin_time=t0)
        t0 += 1.5

        print("last frame ", t0 * FRAME_RATE)

    def solution2b(self):
        cues = self.sub_scenes['solution2b']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # Repeat construction of the question
        title = SimpleTexBObject(r'\text{Solution -- Part 2}', color='important', aligned='center')
        [a, b, c, t] = [0, 1, 0.75 + 1.5j, 0.75]

        # the construction is outsourced
        d_title = 0
        d_problem = 0
        d_part1 = 0
        d_part2 = 0
        dic = self.construction(t0, title,
                                locs=[a, b, c, t],
                                durations=[d_title,
                                           d_problem,
                                           d_part1,
                                           d_part2], text=False)

        t0 += 0.1
        # retrieve variables
        coords = dic['coords']
        sphere_strings = ['s_a', 's_b', 's_c', 's_d', 's_e', 's_t']
        segment_strings = ['l_ab', 'l_ac', 'l_bc', 'l_bd', 'l_de', 'l_tc']
        arc_strings = ['cbd', 'bac']
        displays = ['display0', 'display1']

        spheres = [s_a, s_b, s_c, s_d, s_e, s_t] = get_from_dictionary(dic, sphere_strings)
        segments = [l_ab, l_ac, l_bc, l_bd, l_de, l_tc] = get_from_dictionary(dic, segment_strings)
        [cbd, bac] = get_from_dictionary(dic, arc_strings)

        a_loc = coords.coords2location(z2p(a))
        b_loc = coords.coords2location(z2p(b))
        t_loc = coords.coords2location(z2p(t))
        c_loc = coords.coords2location(z2p(c))
        [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)

        # text work
        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18, flat=False)
        display.set_title(title)
        title.write(transition_time=0)

        colors9 = flatten([['text'] * 5, ['important'], ['text'] * 14, ['important'], ['text'] * 14, ['important'] * 2])
        lines = [
            SimpleTexBObject(r"\text{Coordinates of }D"),
            SimpleTexBObject(r"\text{Line} AC: \mu x+ \mu yi", color='drawing'),
            SimpleTexBObject(r"\mu={2x-1\over x^2+y^2}", color='text'),
            SimpleTexBObject(r"D=\left({(2x-1)\cdot x\over x^2+y^2}\right)+\left({(2x-1)\cdot y\over x^2+y^2}\right)i",
                             name='Line3'),
            SimpleTexBObject(r"(2x-1)\cdot x", name='Line4'),
            SimpleTexBObject(r"2x^2-x", name='Line4b'),
            SimpleTexBObject(r"(2x-1)\cdot y", name='Line6'),
            SimpleTexBObject(r"2xy-y", name='Line6b'),
            SimpleTexBObject(r"A-D=\left({x-2x^2\over x^2+y^2}\right)+\left({y-2xy\over x^2+y^2}\right)i",
                             name='Line8'),
            SimpleTexBObject(r"(A-D)i=\left({x-2x^2\over x^2+y^2}\right)i+\left({y-2xy\over x^2+y^2}\right)\cdot i^2",
                             name='Line9', color=colors9),
            SimpleTexBObject(r"(A-D)i=\left({x-2x^2\over x^2+y^2}\right)i+\left({y-2xy\over x^2+y^2}\right)\cdot (-1)",
                             name='Line9b', color=colors9),
            SimpleTexBObject(
                r"A^\prime=\left({2x^2+2xy-x-y\over x^2+y^2}\right)+\left({-2x^2+2xy+x-y\over x^2+y^2}\right)i",
                name='Line11'),
        ]

        indents = [0.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        rows = [1, 2, 4, 6, 5.64, 5.4, 5.64, 5.6, 10, 12, 12, 14]

        for text, row, indent in zip(lines, rows, indents):
            if row > 1:
                scale = 0.525
            else:
                scale = 0.7
            display.add_text_in(text, line=row, indent=indent, scale=scale)

        lines[3].align(lines[2], char_index=1, other_char_index=1)  # align equal signs
        lines[4].align(lines[3], char_index=1, other_char_index=4)  # align 2s
        lines[5].align(lines[3], char_index=0, other_char_index=4)  # align 2s
        lines[6].align(lines[3], char_index=1, other_char_index=21)  # align 2s
        lines[7].align(lines[3], char_index=0, other_char_index=21)  # align 2s
        lines[8].align(lines[2], char_index=3, other_char_index=1)  # align 2s
        lines[9].align(lines[2], char_index=6, other_char_index=1)  # align 2s
        lines[10].align(lines[2], char_index=6, other_char_index=1)  # align 2s
        lines[11].align(lines[2], char_index=3, other_char_index=1)  # align 2s

        d_zoom = 0
        # zoom in
        coords.zoom(zoom=2, begin_time=t0, transition_time=d_zoom)
        coords.move(direction=[-4, 0, -2], begin_time=t0, transition_time=d_zoom)
        for sphere in spheres:
            sphere.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0, transition_time=d_zoom)
        for seg in segments:
            seg.rescale(rescale=[0.5, 0.5, 1], begin_time=t0, transition_time=d_zoom)

        t0 += 0.1

        # prepare transition
        s_d.write_name_as_label(modus='up', name='D', begin_time=t0 + d_part1 / 4, transition_time=0)
        s_e.write_name_as_label(modus='down', name='E', begin_time=t0 + d_part2 / 4, transition_time=0)
        s_a.write_name_as_label(modus='down_right', name='A=0', begin_time=t0, transition_time=0)
        s_b.write_name_as_label(modus='down_right', name='B=1', begin_time=t0 + 0.3, transition_time=0)
        s_c.write_name_as_label(modus='up', name='C=x+yi', begin_time=t0 + 0.3, transition_time=0)

        l_tc.disappear(begin_time=t0, transition_time=0)
        l_bd.disappear(begin_time=t0, transition_time=0)
        l_bc.disappear(begin_time=t0, transition_time=0)
        s_t.disappear(begin_time=t0, transition_time=0)
        bac.disappear(begin_time=t0, transition_time=0)
        cbd.disappear(begin_time=t0, transition_time=0)

        t0 += 1

        # plot story
        lines[0].write(transition_time=0)

        lines[1].write(begin_time=t0)
        t0 += 1.5
        lines[2].write(begin_time=t0)
        t0 += 1.5

        # construct D
        lines[3].write(letter_set=[0, 1, 2, 17, 18, 19, 34], begin_time=t0)
        t0 += 1

        # real part
        lines[2].move_copy_to(lines[3], src_letter_indices=[2, 5, 8, 10, 6, 3, 4, 7, 9, 11],
                              target_letter_indices=[4, 5, 8, 11, 9, 6, 7, 10, 13, 15], begin_time=t0,
                              offset=[0, 0, -0.001])  # offset that the letters can be nicely overwritten
        t0 += 1
        lines[3].write(letter_set=[9, 3, 12, 14], begin_time=t0, transition_time=0.3)
        t0 += 0.3
        lines[3].write(letter_set=[4, 5, 8, 11, 6, 7, 10, 13, 15], begin_time=t0, transition_time=0)
        lines[3].disappear_copies(begin_time=t0, transition_time=0)

        lines[1].move_copy_to(lines[3], src_letter_indices=[8], target_letter_indices=[16], new_color='text',
                              begin_time=t0, offset=[0, 0.22, -0.001])
        t0 += 1.5

        lines[3].write(letter_set=[16], begin_time=t0, transition_time=0)
        lines[1].disappear_copies(begin_time=t0, transition_time=0)

        # imaginary part
        lines[2].move_copy_to(lines[3], src_letter_indices=[2, 5, 8, 10, 6, 3, 4, 7, 9, 11],
                              target_letter_indices=[21, 22, 25, 28, 26, 23, 24, 27, 30, 32], begin_time=t0,
                              offset=[0, 0, -0.001])  # offset that the letters can be nicely overwritten
        t0 += 1
        lines[3].write(letter_set=[26, 20, 29, 31], begin_time=t0, transition_time=0.3)
        t0 += 0.3
        lines[3].write(letter_set=[21, 22, 25, 28, 26, 23, 24, 27, 30, 32], begin_time=t0, transition_time=0)
        lines[3].disappear_copies(begin_time=t0, transition_time=0)

        lines[1].move_copy_to(lines[3], src_letter_indices=[11], target_letter_indices=[33], new_color='text',
                              begin_time=t0, offset=[0, 0.22, -0.001])
        lines[1].disappear_copies(begin_time=t0 + 1.5, transition_time=0)
        lines[1].move_copy_to(lines[3], src_letter_indices=[12], target_letter_indices=[35], new_color='text',
                              begin_time=t0, offset=[0, 0, -0.001])
        lines[1].disappear_copies(begin_time=t0 + 1.5, transition_time=0)
        t0 += 1.5
        lines[3].write(letter_set=[33, 35], begin_time=t0, transition_time=0)

        # simplify D
        t0 += 0.1
        ibpy.disappear_all_copies_of_letters(begin_time=t0, transition_time=0)
        lines[4].write(begin_time=t0, transition_time=0)
        lines[6].write(begin_time=t0, transition_time=0)
        set = [3, 4, 5, 8, 11, 12, 14, 16]
        for l in set:
            lines[3].letters[l].disappear(begin_time=t0, transition_time=0.1)
        set = [20, 21, 22, 25, 28, 29, 31, 33]
        for l in set:
            lines[3].letters[l].disappear(begin_time=t0, transition_time=0.1)
        t0 += 0.2
        lines[4].replace(lines[5], begin_time=t0)
        t0 += 1.5
        lines[6].replace(lines[7], begin_time=t0)
        t0 += 1

        addon = SimpleTexBObject(r"\text{Coordinates of }A^\prime")
        display.add_text_in(addon, line=8, indent=0.5)
        addon.write(begin_time=t0)
        t0 += 1.5

        # preview rotation

        da = a_loc - d_loc
        s_ap = Sphere(0.24, location=da, name="A^\prime", label_rotation=[np.pi / 2, np.pi / 2, 0])
        s_ap.grow(begin_time=t0, transition_time=0, scale=0.5)

        l_da = Cylinder.from_start_to_end(start=Vector(), end=da,
                                          color='drawing', thickness=0.26, name='lbc')
        l_da.grow(modus='from_start', begin_time=t0, transition_time=0)
        t0 += 1.5

        rot_box = BObject(children=[s_ap, l_da], location=d_loc)
        rot_box.appear(begin_time=t0, transition_time=0)
        rot_box.rotate(rotation_euler=[0, -np.pi / 2, 0], begin_time=t0)
        coords.add_object(rot_box)

        t0 += 1
        s_ap.write_name_as_label(begin_time=t0, modus='down_left')
        t0 += 1.5

        # rot_box.disappear(begin_time=t0)
        l_de.disappear(begin_time=t0)
        s_e.disappear(begin_time=t0)
        t0 += 1.5

        # shift D

        arrows = [
            PArrow(start=d_loc, end=Vector(), name='Arrow1', color='example', thickness=1),
            PArrow(start=d_loc, end=Vector(), name='Arrow2', color='example', thickness=1),
        ]

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=1)

        t0 += 1.5

        end_positions = [Vector(), da]

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]

        for container in containers:
            container.appear(begin_time=t0)

        s_m = Sphere(0.2, location=d_loc, color='text', name='D-D')
        s_m2 = Sphere(0.2, location=a_loc, color='text', name='A-D')

        spheres = [s_m2, s_m]
        for s in spheres:
            s.grow(scale=0.5, begin_time=t0, transition_time=0)

        modi = ['up_left', 'up_left']
        for modus, sphere in zip(modi, spheres):
            sphere.move(direction=da, begin_time=t0)
            sphere.write_name_as_label(modus=modus, begin_time=t0)

        for container in containers:
            container.shrink(begin_time=t0)

        t0 += 1.5

        # write real part of A-D
        lines[8].write(letter_set=[0, 1, 2, 3, 4, 16, 17, 18, 30, 31], begin_time=t0)
        t0 += 1
        lines[4].move_copy_to(lines[8], src_letter_indices=[5], target_letter_indices=[5], begin_time=t0,
                              offset=[0, 0.26, -0.001])
        t0 += 1
        lines[8].write(letter_set=[5, 8], begin_time=t0, transition_time=0)
        lines[4].disappear_copies(begin_time=t0)
        t0 += 1.5
        lines[4].move_copy_to(lines[8], src_letter_indices=[1, 2, 3], target_letter_indices=[11, 12, 14], begin_time=t0,
                              offset=[0, 0.26, -0.001])
        t0 += 1
        lines[8].write(letter_set=[11, 12, 14], begin_time=t0, transition_time=0)
        lines[4].disappear_copies(begin_time=t0)
        t0 += 1.5
        lines[3].move_copy_to(lines[8], src_letter_indices=[6, 7, 9, 10, 13, 15],
                              target_letter_indices=[6, 7, 9, 10, 13, 15], begin_time=t0,
                              offset=[0, 0, -0.001])
        t0 += 1
        lines[8].write(letter_set=[6, 7, 9, 10, 13, 15], begin_time=t0, transition_time=0)
        lines[3].disappear_copies(begin_time=t0)
        t0 += 1.5

        # write imaginary part of A-D
        lines[7].write(begin_time=t0, transition_time=0)
        lines[6].disappear(begin_time=t0, transition_time=0)
        lines[7].move_copy_to(lines[8], src_letter_indices=[4, 0, 1, 2, 3]
                              , target_letter_indices=[19, 25, 26, 28, 22], begin_time=t0,
                              offset=[0, 0.26, -0.001])
        t0 += 1
        lines[8].write(letter_set=[19, 25, 26, 28], begin_time=t0, transition_time=0)
        lines[6].disappear_copies(begin_time=t0)
        t0 += 1.5
        lines[3].move_copy_to(lines[8], src_letter_indices=[23, 24, 26, 27, 30, 32],
                              target_letter_indices=[20, 21, 23, 24, 27, 29], begin_time=t0,
                              offset=[0, 0, -0.001])
        t0 += 1
        lines[8].write(letter_set=[20, 21, 22, 23, 24, 27, 29], begin_time=t0, transition_time=0)
        lines[3].disappear_copies(begin_time=t0)
        t0 += 1.5

        # rotate A-D

        lines[9].write(letter_set=[0, 1, 2, 3, 4, 5, 6, 7, 19, 21, 22, 34], begin_time=t0)
        t0 += 1
        lines[8].move_copy_to_and_remove(lines[9], src_letter_indices=[5, 8, 11, 12, 14, 9, 6, 7, 10, 13, 15],
                                         target_letter_indices=[8, 11, 14, 15, 17, 12, 9, 10, 13, 16, 18],
                                         begin_time=t0)
        lines[9].move_copy_to_and_remove(src_letter_indices=[5], target_letter_indices=[20],
                                         new_color='important', begin_time=t0)
        t0 += 1.5
        lines[8].move_copy_to_and_remove(lines[9], src_letter_indices=[19, 22, 25, 26, 28, 23, 20, 21, 24, 27, 29, 31],
                                         target_letter_indices=[23, 26, 29, 30, 32, 27, 24, 25, 28, 31, 33, 36],
                                         begin_time=t0)
        lines[9].move_copy_to_and_remove(src_letter_indices=[5], target_letter_indices=[36], new_color='important',
                                         begin_time=t0)
        lines[9].write(letter_set=[35, 37], begin_time=t0 + 1, transition_time=0)  # write power of i
        t0 += 1.5
        lines[9].replace(lines[10], src_letter_range=[36, 38], img_letter_range=[36, 40], begin_time=t0)
        t0 += 1
        lines[10].write(begin_time=t0, transition_time=0)
        lines[9].disappear(begin_time=t0)
        t0 += 1

        s_m3 = Sphere(0.2, location=da, color='text', name='(A-D)i', label_rotation=[np.pi / 2, np.pi / 2, 0])
        s_m3.grow(scale=0.5, begin_time=t0, transition_time=0)
        rot_box2 = BObject(children=[s_m3], location=Vector())
        rot_box2.appear(begin_time=t0, transition_time=0)
        rot_box2.rotate(rotation_euler=[0, -np.pi / 2, 0], begin_time=t0)
        t0 += 1

        s_m3.write_name_as_label(modus='down_left', begin_time=t0)
        t0 += 1.5

        # shift back

        arrows2 = [
            PArrow(start=da, end=Vector(), name='Arrow3', color='example', thickness=1),
            PArrow(start=da, end=Vector(), name='Arrow4', color='example', thickness=1),
            PArrow(start=da, end=Vector(), name='Arrow5', color='example', thickness=1),
        ]

        for arrow in arrows2:
            arrow.grow(begin_time=t0, transition_time=1)

        t0 += 1.5
        ap_loc = Vector([-da.z, 0, da.x]) - da
        end_positions2 = [Vector(), -da, ap_loc]

        containers2 = [BObject(children=[arrows2[i]], location=end_positions2[i]) for i in range(0, len(arrows2))]

        for container in containers2:
            container.appear(begin_time=t0)

        s_m4 = Sphere(0.2, location=ap_loc + da, name='(A-D)i+D')
        s_m4.grow(scale=0.5, begin_time=t0, transition_time=0)
        spheres = [s_m2, s_m, s_m4]
        for sphere in spheres:
            sphere.move(direction=-da, begin_time=t0)

        s_m2.label_disappear(begin_time=t0)
        s_m.label_disappear(begin_time=t0)
        s_m4.write_name_as_label(modus='up_right', begin_time=t0, aligned='left')

        for container in containers2:
            container.shrink(begin_time=t0)
        t0 += 1.5

        # write A'
        lines[11].write(letter_set=[0, 1, 2, 3, 8, 10, 13, 14, 16, 17, 21, 22, 23, 29, 31, 33, 34, 36, 38, 42, 43],
                        begin_time=t0)
        t0 += 1.5

        # real part of A-D
        lines[10].move_copy_to_and_remove(lines[11], src_letter_indices=[23],
                                          target_letter_indices=[20], begin_time=t0,
                                          offset=[0, 0, -0.001])
        lines[10].move_copy_to_and_remove(lines[11], src_letter_indices=[37],
                                          target_letter_indices=[19], begin_time=t0,
                                          offset=[0, 0.26, -0.001])
        t0 += 1

        lines[10].move_copy_to_and_remove(lines[11], src_letter_indices=[26, 29, 30, 32],
                                          target_letter_indices=[7, 9, 11, 12], begin_time=t0,
                                          offset=[0, 0, -0.001])
        lines[10].move_copy_to_and_remove(lines[11], src_letter_indices=[37],
                                          target_letter_indices=[7], begin_time=t0,
                                          offset=[0, 0.26, -0.001])
        lines[11].write(letter_set=[7], begin_time=t0 + 1, transition_time=0)  # add plus sign
        t0 += 1

        # imag part of A-D
        lines[10].move_copy_to(lines[11], src_letter_indices=[11, 14, 15, 17, 20, 8],
                               target_letter_indices=[24, 25, 26, 27, 43, 39], begin_time=t0,
                               offset=[0, 0, -0.001])
        set = [24, 25, 26, 27, 39]  # write all target letter except the i, which has already been written
        lines[11].write(letter_set=set, begin_time=t0 + 1, transition_time=0)
        lines[10].disappear_copies(begin_time=t0 + 1)

        lines[11].write(letter_set=[37], begin_time=t0 + 1, transition_time=0)  # add plus sign
        t0 += 1.5

        # real part D
        lines[4].move_copy_to_and_remove(lines[11], src_letter_indices=[1, 2, 3, 4, 5],
                                         target_letter_indices=[4, 5, 6, 15, 18], begin_time=t0,
                                         offset=[0, 0.26, -0.001])
        t0 += 1.5
        # imag part D
        lines[7].move_copy_to_and_remove(lines[11], src_letter_indices=[0, 1, 2, 3, 4]
                                         , target_letter_indices=[30, 32, 35, 40, 41], begin_time=t0,
                                         offset=[0, 0.26, -0.001])
        lines[11].write(letter_set=[28], begin_time=t0 + 1, transition_time=0)  # add plus sign
        lines[3].move_copy_to(lines[11], src_letter_indices=[35], target_letter_indices=[43],
                              begin_time=t0, offset=[0, 0, -0.001])
        lines[3].disappear_copies(begin_time=t0 + 1)
        t0 += 1.5

        title_back = SimpleTexBObject(r'\text{Solution -- Part 2}', color='important', aligned='center')
        display.set_title_back(title_back)
        title_back.write(begin_time=t0, transition_time=0)

        lines_back = [
            SimpleTexBObject(r"D=\left({2x^2-x\over x^2+y^2}\right)+\left({2xy-y\over x^2+y^2}\right)i",
                             name='Lineb0'),
            SimpleTexBObject(
                r"A^\prime=\left({2x^2+2xy-x-y\over x^2+y^2}\right)+\left({-2x^2+2xy+x-y\over x^2+y^2}\right)i",
                name='Lineb1'),

        ]
        indents_back = [1.6, 1.5]
        rows_back = [2, 4]

        for text, row, indent in zip(lines_back, rows_back, indents_back):
            scale = 0.5
            display.add_text_in_back(text, line=row, indent=indent, scale=scale)

        lines_back[1].align(lines_back[0], char_index=2, other_char_index=1)

        for text in lines_back:
            text.write(begin_time=t0, transition_time=0)

        display.turn(begin_time=t0)

        s_m4.disappear(begin_time=t0)
        s_m3.disappear(begin_time=t0)

        t0 += 2

        coords.add_objects([rot_box2, s_m4])
        coords.add_objects(containers)
        coords.add_objects(containers2)
        coords.add_objects(spheres)

        print("last frame ", t0 * FRAME_RATE)

    def solution2c(self):
        cues = self.sub_scenes['solution2c']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # Repeat construction of the question
        title = SimpleTexBObject(r'\text{Solution -- Part 2}', color='important', aligned='center')
        [a, b, c, t] = [0, 1, 0.75 + 1.5j, 0.75]

        # the construction is outsourced
        d_title = 0
        d_problem = 0
        d_part1 = 0
        d_part2 = 0
        dic = self.construction(t0, title,
                                locs=[a, b, c, t],
                                durations=[d_title,
                                           d_problem,
                                           d_part1,
                                           d_part2], text=False)

        t0 += 0.1
        # retrieve variables
        coords = dic['coords']
        sphere_strings = ['s_a', 's_b', 's_c', 's_d', 's_e', 's_t']
        segment_strings = ['l_ab', 'l_ac', 'l_bc', 'l_bd', 'l_de', 'l_tc']
        arc_strings = ['cbd', 'bac']
        displays = ['display0', 'display1']

        spheres = [s_a, s_b, s_c, s_d, s_e, s_t] = get_from_dictionary(dic, sphere_strings)
        segments = [l_ab, l_ac, l_bc, l_bd, l_de, l_tc] = get_from_dictionary(dic, segment_strings)
        [cbd, bac] = get_from_dictionary(dic, arc_strings)

        a_loc = coords.coords2location(z2p(a))
        b_loc = coords.coords2location(z2p(b))
        t_loc = coords.coords2location(z2p(t))
        c_loc = coords.coords2location(z2p(c))
        [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)

        # text work
        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18, flat=False)
        display.set_title(title)
        title.write(transition_time=0)

        colors = flatten(
            [['example'], ['text'] * 10, ['example'], ['text'] * 8, ['example'], ['text'] * 8, ['example'],
             ['text'] * 8,
             ['example'], ['text']])
        lines = [
            SimpleTexBObject(r"D=\left({2x^2-x\over x^2+y^2}\right)+\left({2xy-y\over x^2+y^2}\right)i",
                             name='Line0'),
            SimpleTexBObject(
                r"A^\prime=\left({2x^2+2xy-x-y\over x^2+y^2}\right)+\left({-2x^2+2xy+x-y\over x^2+y^2}\right)i",
                name='Line1'),
            SimpleTexBObject(r"\text{Line } \,\,DA^\prime", color='example', name='Line2'),
            SimpleTexBObject(r"A^\prime-D=\left({2xy-y\over x^2+y^2}\right)+\left({-2x^2+x\over x^2+y^2}\right)i",
                             name='Line3'),
            SimpleTexBObject(
                r"\zeta\cdot(A^\prime-D)=\left({2\zeta xy-\zeta y\over x^2+y^2}\right)+\left({-2\zeta x^2+\zeta x\over x^2+y^2}\right)i",
                name='Line4', color=colors),
            SimpleTexBObject(
                r"DA^\prime: \left({2\zeta xy-\zeta y+2x^2-x\over x^2+y^2}\right)+\left({-2\zeta x^2+\zeta x+2xy-y\over x^2+y^2}\right)i",
                name='Line5', color='example'),
        ]
        indents = [1.6, 1.5, 0.5, 1, 1, 0.5]
        rows = [2, 4, 6, 8, 10, 14]

        for text, row, indent in zip(lines, rows, indents):
            if row != 6:
                scale = 0.5
            else:
                scale = 0.7
            display.add_text_in(text, line=row, indent=indent, scale=scale)

        lines[1].align(lines[0], char_index=2, other_char_index=1)
        lines[3].align(lines[1], char_index=4, other_char_index=2)
        lines[4].align(lines[1], char_index=8, other_char_index=2)

        d_zoom = 0
        # zoom in
        coords.zoom(zoom=2, begin_time=t0, transition_time=d_zoom)
        coords.move(direction=[-4, 0, -2], begin_time=t0, transition_time=d_zoom)
        for sphere in spheres:
            sphere.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0, transition_time=d_zoom)
        for seg in segments:
            seg.rescale(rescale=[0.5, 0.5, 1], begin_time=t0, transition_time=d_zoom)

        t0 += 0.1

        # prepare transition
        s_d.write_name_as_label(modus='up', name='D', begin_time=t0 + d_part1 / 4, transition_time=0)
        s_a.write_name_as_label(modus='down_right', name='A=0', begin_time=t0, transition_time=0)
        s_b.write_name_as_label(modus='down_right', name='B=1', begin_time=t0 + 0.3, transition_time=0)
        s_c.write_name_as_label(modus='up', name='C=x+yi', begin_time=t0 + 0.3, transition_time=0)

        l_tc.disappear(begin_time=t0, transition_time=0)
        l_bd.disappear(begin_time=t0, transition_time=0)
        l_bc.disappear(begin_time=t0, transition_time=0)
        l_de.disappear(transition_time=0)
        s_t.disappear(begin_time=t0, transition_time=0)
        bac.disappear(begin_time=t0, transition_time=0)
        cbd.disappear(begin_time=t0, transition_time=0)
        s_e.disappear(transition_time=0)

        t0 += 1

        da = a_loc - d_loc
        ap_loc = Vector([-da.z, 0, da.x]) - da
        s_ap = Sphere(0.24, location=ap_loc, name="A^\prime")
        s_ap.grow(begin_time=t0, transition_time=0, scale=0.5)
        s_ap.write_name_as_label(modus='down_right', transition_time=0)

        l_dap = Cylinder.from_start_to_end(start=d_loc, end=ap_loc,
                                           color='drawing', thickness=0.26, name='lbc')
        l_dap.grow(modus='from_start', begin_time=t0, transition_time=0)

        lines[0].write(transition_time=0)
        lines[1].write(transition_time=0)

        # plot story

        lines[2].write(begin_time=t0)
        t0 += 1.5

        # construct line
        duration = 15  # this way the drawing steps are 1.5 s intervals or 3 s intervals
        self.construct_line(coords, d_loc, 'D', ap_loc, r'A^\prime', t0, duration, r'\zeta', 'example', 2, 6, 4)

        t0 += 3.5  # shift by -D has happened

        set = [4, 5, 6, 15, 18]
        for i in set:
            lines[1].letters[i].change_color(new_color='joker', begin_time=t0)

        set = [3, 5, 7, 10, 12]
        for i in set:
            lines[0].letters[i].change_color(new_color='joker', begin_time=t0)

        t0 += 1

        set = [30, 32, 35, 40, 41]
        for i in set:
            lines[1].letters[i].change_color(new_color='important', begin_time=t0)

        set = [17, 19, 21, 24, 26]
        for i in set:
            lines[0].letters[i].change_color(new_color='important', begin_time=t0)

        t0 += 1.5

        lines[3].write(letter_set=[0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 14, 16, 17, 18, 19, 21, 23, 24, 26, 29, 30, 32, 33],
                       begin_time=t0)

        t0 += 1.5

        lines[1].move_copy_to_and_remove(lines[3], src_letter_indices=[9, 11, 12, 19, 20]
                                         , target_letter_indices=[6, 8, 10, 13, 15], begin_time=t0,
                                         offset=[0, 0, -0.001])

        t0 += 1.5

        lines[1].move_copy_to_and_remove(lines[3], src_letter_indices=[24, 25, 26, 27, 37, 39]
                                         , target_letter_indices=[20, 22, 24, 25, 28, 31], begin_time=t0,
                                         offset=[0, 0, -0.001])

        t0 += 1.5

        t0 += 4  # drawing of the line

        lines[4].write(
            letter_set=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 16, 17, 19, 21, 23, 24, 25, 28, 30, 33, 34, 36, 37, 40,
                        41], begin_time=t0)
        t0 += 1.5
        lines[3].move_copy_to_and_remove(lines[4], src_letter_indices=[6, 8, 10],
                                         target_letter_indices=[10, 13, 15], offset=[0, 0, -0.001],
                                         begin_time=t0)
        lines[4].move_copy_to_and_remove(lines[4], src_letter_indices=[0],
                                         target_letter_indices=[11],
                                         offset=[0, 0.22, -0.001],
                                         begin_time=t0)
        t0 += 1
        lines[3].move_copy_to_and_remove(lines[4], src_letter_indices=[13, 15],
                                         target_letter_indices=[18, 22], offset=[0, 0, -0.001],
                                         begin_time=t0)
        lines[4].move_copy_to_and_remove(lines[4], src_letter_indices=[0],
                                         target_letter_indices=[20],
                                         offset=[0, 0.22, -0.001],
                                         begin_time=t0)
        t0 += 1
        lines[3].move_copy_to_and_remove(lines[4], src_letter_indices=[20, 22, 24, 25],
                                         target_letter_indices=[26, 27, 31, 32], offset=[0, 0, -0.001],
                                         begin_time=t0)
        lines[4].move_copy_to_and_remove(lines[4], src_letter_indices=[0],
                                         target_letter_indices=[29],
                                         offset=[0, 0.22, -0.001],
                                         begin_time=t0)
        t0 += 1
        lines[3].move_copy_to_and_remove(lines[4], src_letter_indices=[28, 31],
                                         target_letter_indices=[35, 39], offset=[0, 0, -0.001],
                                         begin_time=t0)
        lines[4].move_copy_to_and_remove(lines[4], src_letter_indices=[0],
                                         target_letter_indices=[38],
                                         offset=[0, 0.22, -0.001],
                                         begin_time=t0)
        t0 += 1.5

        t0 += 4  # shifting back

        lines[5].write(letter_set=[0, 1, 2, 3], begin_time=t0, transition_time=0.3)
        t0 += 0.5
        lines[4].move_copy_to_and_remove(lines[5], src_letter_indices=range(9, 24),
                                         target_letter_indices=[4, 5, 6, 10, 7, 12, 8, 14, 15, 9, 17, 11, 19, 13, 24],
                                         offset=[0, 0, -0.001],
                                         begin_time=t0, new_color='example')
        t0 += 1
        lines[4].move_copy_to_and_remove(lines[5], src_letter_indices=range(24, 42),
                                         target_letter_indices=[25, 26, 27, 28, 33, 29, 34, 30, 31, 37, 38, 32, 40, 41,
                                                                35, 36, 47, 48], offset=[0, 0, -0.001],
                                         begin_time=t0, new_color='example')
        t0 += 1
        lines[5].write(letter_set=[16], begin_time=t0, transition_time=0)
        lines[0].move_copy_to_and_remove(lines[5], src_letter_indices=[3, 5, 7, 10, 12],
                                         target_letter_indices=[18, 20, 21, 22, 23], offset=[0, 0, -0.001],
                                         begin_time=t0, new_color='example')

        t0 += 1
        lines[5].write(letter_set=[39], begin_time=t0, transition_time=0)
        lines[0].move_copy_to_and_remove(lines[5], src_letter_indices=[17, 19, 21, 24, 26, 29],
                                         target_letter_indices=[42, 43, 44, 45, 46, 48], offset=[0, 0, -0.001],
                                         begin_time=t0, new_color='example')
        t0 += 3

        s_ap.disappear(begin_time=t0)
        l_dap.disappear(begin_time=t0)

        title_back = SimpleTexBObject(r'\text{Solution -- Part 2}', color='important', aligned='center')
        display.set_title_back(title_back)
        title_back.write(begin_time=t0, transition_time=0)

        lines_back = [
            SimpleTexBObject(
                r"DA^\prime: \left({2\zeta xy-\zeta y+2x^2-x\over x^2+y^2}\right)+\left({-2\zeta x^2+\zeta x+2xy-y\over x^2+y^2}\right)i",
                name='Line0', color='example'),
        ]
        indents_back = [0.5]
        rows_back = [2]

        for text, row, indent in zip(lines_back, rows_back, indents_back):
            scale = 0.5
            display.add_text_in_back(text, line=row, indent=indent, scale=scale)
            text.write(begin_time=t0, transition_time=0)

        display.turn(begin_time=t0)

        coords.add_objects([s_ap, l_dap])
        print("last frame ", t0 * FRAME_RATE)

    def construct_line(self, coords, a_loc, name_a, b_loc, name_b, t0, duration, param, color, zoom, break1, break2):
        # shift to origin
        steps = 10
        dt = 0.8 * duration / steps
        sep = 0.2 * duration / steps

        arrows = [
            PArrow(start=a_loc, end=Vector(), name='Arrow1', color='example', thickness=2 / zoom),
            PArrow(start=a_loc, end=Vector(), name='Arrow2', color='example', thickness=2 / zoom),
        ]

        coords.add_objects(arrows)

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=dt)

        t0 += dt + sep

        end_positions = [b_loc - a_loc, Vector()]

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]
        coords.add_objects(containers)

        for container in containers:
            container.appear(begin_time=t0)

        s_m = Sphere(0.2, location=b_loc, color='text', name=name_b + "-" + name_a)
        s_m2 = Sphere(0.2, location=a_loc, color='text', name=name_a + '-' + name_a)

        spheres = [s_m2, s_m]
        modi = ['down_left', 'down_left']
        for modus, sphere in zip(modi, spheres):
            sphere.grow(scale=1 / zoom, begin_time=t0, transition_time=0)
            sphere.move(direction=-a_loc, begin_time=t0, transition_time=2 * dt)
            sphere.write_name_as_label(modus=modus, begin_time=t0, transition_time=2 * dt)
        for container in containers:
            container.shrink(begin_time=t0, transition_time=2 * dt)
        t0 += 2 * (dt + sep) + break1

        # draw param line

        v = b_loc - a_loc
        tracer = Sphere(0.1, location=-1.99 * v, color=color)
        tracer.grow(scale=0.5, begin_time=t0, transition_time=dt / 2)
        t0 += dt / 2 + sep / 2

        line = Cylinder.from_start_to_end(start=-2 * v, end=4.1 * v, color='text', thickness=0.5 / zoom)
        line.grow(modus='from_start', begin_time=t0, transition_time=3.5 * dt)
        tracer.move(direction=6.1 * v, begin_time=t0, transition_time=3.5 * dt)

        l_old = -np.Infinity
        labels = []
        movers = [line]
        removables = []
        count = 0

        pos = flatten([['up_right'] * 6, ['down_right'], ['up_right']])
        for i in range(-2, 6):
            labels.append(param + "=" + str(i))
        for frame in range(int(t0 * FRAME_RATE), int((t0 + 3.5 * dt) * FRAME_RATE) + 1):
            location = ibpy.get_location_at_frame(tracer, frame)
            l = np.floor(- np.sign(location.x) * location.length / v.length)
            if l_old != l:
                l_old = l
                sphere = Sphere(0.15, location=location, name=labels[count], color='text')
                t = frame / FRAME_RATE
                sphere.grow(scale=1 / zoom, begin_time=t, transition_time=0.1)
                count += 1
                sphere.write_name_as_label(aligned='left', modus=pos[count], begin_time=t + 0.1, transition_time=0.2)
                movers.append(sphere)
                removables.append(sphere)
                coords.add_object(sphere)

        coords.add_objects([s_m2, s_m, line, tracer])

        t0 += 3.5 * (dt + sep) + break2

        # shift back

        arrows = [
            PArrow(start=-a_loc, end=Vector(), name='Arrow3', color='example', thickness=2 / zoom),
            PArrow(start=-a_loc, end=Vector(), name='Arrow4', color='example', thickness=2 / zoom),
        ]

        end_positions = [a_loc, b_loc]

        for i in range(1, len(movers)):
            arrows.append(
                PArrow(start=-a_loc, end=Vector(), name='Arrow' + str(i + 4), color='example', thickness=2 / zoom),
            )
            end_positions.append((i - 3) * v + a_loc)

        coords.add_objects(arrows)

        for arrow in arrows:
            arrow.grow(begin_time=t0, transition_time=2 * dt)

        t0 += 2 * (dt + sep)

        containers = [BObject(children=[arrows[i]], location=end_positions[i]) for i in range(0, len(arrows))]
        coords.add_objects(containers)

        for container in containers:
            container.appear(begin_time=t0)

        for sphere in spheres:
            sphere.move(direction=a_loc, begin_time=t0, transition_time=dt)
            sphere.label_disappear(begin_time=t0)

        for mover in movers:
            mover.move(direction=a_loc, begin_time=t0, transition_time=dt)
            mover.change_color(new_color=color, begin_time=t0, transition_time=dt)
            if mover.label:
                mover.label.change_color(new_color=color, begin_time=t0, transition_time=dt)

        for container in containers:
            container.shrink(begin_time=t0)

        tracer.disappear(begin_time=t0, transition_time=dt)
        t0 += dt + sep

        for i in range(1, len(movers)):
            movers[i].disappear(begin_time=t0)

        s_m.disappear(begin_time=t0)
        s_m2.disappear(begin_time=t0)

    def solution2d(self):
        cues = self.sub_scenes['solution2d']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # Repeat construction of the question
        title = SimpleTexBObject(r'\text{Solution -- Part 2}', color='important', aligned='center')
        [a, b, c, t] = [0, 1, 0.75 + 1.5j, 0.75]

        # the construction is outsourced
        d_title = 0
        d_problem = 0
        d_part1 = 0
        d_part2 = 0
        dic = self.construction(t0, title,
                                locs=[a, b, c, t],
                                durations=[d_title,
                                           d_problem,
                                           d_part1,
                                           d_part2], text=False)

        t0 += 0.1
        # retrieve variables
        coords = dic['coords']
        sphere_strings = ['s_a', 's_b', 's_c', 's_d', 's_e', 's_t']
        segment_strings = ['l_ab', 'l_ac', 'l_bc', 'l_bd', 'l_de', 'l_tc']
        arc_strings = ['cbd', 'bac','ra','ra2']

        spheres = [s_a, s_b, s_c, s_d, s_e, s_t] = get_from_dictionary(dic, sphere_strings)
        segments = [l_ab, l_ac, l_bc, l_bd, l_de, l_tc] = get_from_dictionary(dic, segment_strings)
        [cbd, bac,ra,ra2] = get_from_dictionary(dic, arc_strings)

        a_loc = coords.coords2location(z2p(a))
        b_loc = coords.coords2location(z2p(b))
        t_loc = coords.coords2location(z2p(t))
        c_loc = coords.coords2location(z2p(c))
        [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)

        # text work
        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18, flat=False)
        display.set_title(title)
        title.write(transition_time=0)

        colors = flatten([['important'] * 2, ['text']])
        lines = [
            SimpleTexBObject(
                r"DA^\prime: \left({2\zeta xy-\zeta y+2x^2-x\over x^2+y^2}\right)+\left({-2\zeta x^2+\zeta x+2xy-y\over x^2+y^2}\right)i",
                name='Line0', color='example'),
            SimpleTexBObject(r'\text{Coordinates of }E:', name='Line1',),
            SimpleTexBObject(r"{-2\zeta x^2+\zeta x+2xy-y\over x^2+y^2}=0", name='Line2', color='text',bevel=0),
            SimpleTexBObject(r"-2\zeta x^2+\zeta x+2xy-y=0", name='Line3', color='text',bevel=0),
            SimpleTexBObject(r"-2\zeta x^2+\zeta x-2xy+y=0", name='Line3b', color='text',bevel=0),  # replace signs
            SimpleTexBObject(r"-2\zeta x^2+\zeta x-2xy+y=-2xy+y", name='Line5', color='text',bevel=0),
            SimpleTexBObject(r"\zeta(-2 x^2+x)-2xy+y=-2xy+y", name='Line6', color='text',bevel=0),
            SimpleTexBObject(r"\zeta(-2 x^2+x)-2xy+y={-2xy+y\over(-2 x^2+x)}", name='Line7', color='text',bevel=0),
            SimpleTexBObject(r"-\! 2xy+y", name='Line8', color='text',bevel=0),
            SimpleTexBObject(r"(-2 x^2+x)", name='Line9', color='text',bevel=0),
            SimpleTexBObject(r"\zeta={y\over x}", name='Line10', color='text',bevel=0),
            SimpleTexBObject(r"y(-2x+1)", name='Line11', color='text',bevel=0),
            SimpleTexBObject(r"x(-2x+1)", name='Line12', color='text',bevel=0),
            SimpleTexBObject(
                r"E=\left({2\tfrac{y}{x} xy-\tfrac{y}{x} y+2x^2-x\over x^2+y^2}\right)+\left({-2{y\over x} x^2+{y\over x} x+2xy-y\over x^2+y^2}\right)i",
                name='Line13', color=colors,bevel=0),
            SimpleTexBObject(r"-\!2\tfrac{y}{x}x^2+\tfrac{y}{x}x+2xy-y", name='Line14',bevel=0),
            SimpleTexBObject(r"-2xy+y+2xy-y", name='Line15',bevel=0),
            SimpleTexBObject(r"E={\left(x^2+y^2\right)\cdot\left(1-\tfrac{1}{x}\right)\over x^2+y^2}", name='Line16',
                             color=colors,bevel=0),
            SimpleTexBObject(r"2y^2-\tfrac{y^2}{x}", name='Line17',bevel=0),
            SimpleTexBObject(r"+", name='Line18',bevel=0),
            SimpleTexBObject(r"2x^2-x", name='Line19',bevel=0),
            SimpleTexBObject(r"y^2\left(2-\tfrac{1}{x}\right)", name='Line17b',bevel=0),
            SimpleTexBObject(r"x^2\left(2-\tfrac{1}{x}\right)", name='Line19b',bevel=0),
            SimpleTexBObject(r"\text{\textcircled{$\checkmark$}}", color='important', name='Check'),
            SimpleTexBObject(r'\text{$E$ does not depend on $y$}', color='important',name='Line23',bevel=0),
            SimpleTexBObject(r"\text{\fbox{\phantom{\text{$E$ does not depend on $y$}}}}",color='important',name='Line24',bevel=0)
        ]
        indents = [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0.5, 1, 1, 1, 1.7, 4, 4.5,
                   1.7, 4.2, 9,0.5,0.5]
        rows = [2, 4, 6, 8, 8, 8, 8, 8.075, 7.6, 8.48,
                8.075, 7.6, 8.48, 11, 10.65625, 10.65625, 13, 12.5, 12.6, 12.45,
                12.5, 12.45, 15.5,15,15]
        scales = [0.5, 0.7, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                  0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7,
                  0.7, 0.7, 1.4,0.7,0.7]
        for text, row, indent, scale in zip(lines, rows, indents, scales):
            display.add_text_in(text, line=row, indent=indent, scale=scale)

        lines[3].align(lines[2], char_index=14, other_char_index=20)
        lines[4].align(lines[2], char_index=14, other_char_index=20)
        lines[5].align(lines[2], char_index=14, other_char_index=20)
        lines[6].align(lines[2], char_index=15, other_char_index=20)
        lines[7].align(lines[2], char_index=15, other_char_index=20)
        lines[8].align(lines[7], char_index=0, other_char_index=17)
        lines[9].align(lines[7], char_index=0, other_char_index=16)
        lines[10].align(lines[2], char_index=1, other_char_index=20)
        lines[11].align(lines[10], char_index=0, other_char_index=4)
        lines[12].align(lines[10], char_index=0, other_char_index=3)
        lines[14].align(lines[13], char_index=0, other_char_index=29)
        lines[15].align(lines[14], char_index=11, other_char_index=17)
        lines[16].align(lines[13], char_index=2, other_char_index=2)

        d_zoom = 0
        # zoom in
        coords.zoom(zoom=2, begin_time=t0, transition_time=d_zoom)
        coords.move(direction=[-4, 0, -2], begin_time=t0, transition_time=d_zoom)
        for sphere in spheres:
            sphere.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0, transition_time=d_zoom)
        for seg in segments:
            seg.rescale(rescale=[0.5, 0.5, 1], begin_time=t0, transition_time=d_zoom)

        t0 += 0.1

        # prepare transition
        s_d.write_name_as_label(modus='up', name='D', begin_time=t0 + d_part1 / 4, transition_time=0)
        s_a.write_name_as_label(modus='down_right', name='A=0', begin_time=t0, transition_time=0)
        s_b.write_name_as_label(modus='down_right', name='B=1', begin_time=t0 + 0.3, transition_time=0)
        s_c.write_name_as_label(modus='up', name='C=x+yi', begin_time=t0 + 0.3, transition_time=0)

        l_tc.disappear(begin_time=t0, transition_time=0)
        l_bd.disappear(begin_time=t0, transition_time=0)
        l_bc.disappear(begin_time=t0, transition_time=0)
        l_de.disappear(transition_time=0)
        s_t.disappear(begin_time=t0, transition_time=0)
        bac.disappear(begin_time=t0, transition_time=0)
        cbd.disappear(begin_time=t0, transition_time=0)
        s_e.disappear(transition_time=0, alpha=0.001)
        ra.disappear(begin_time=t0,transition_time=0)
        ra2.disappear(begin_time=t0,transition_time=0)

        t0 += 1

        da = a_loc - d_loc
        ap_loc = Vector([-da.z, 0, da.x]) - da
        v = ap_loc - d_loc

        l_de_long = Cylinder.from_start_to_end(start=d_loc - 2 * v, end=ap_loc + 3 * v,
                                               color='example', thickness=0.26, name='l_de')
        l_de_long.grow(modus='from_start', begin_time=t0, transition_time=0)

        lines[0].write(transition_time=0)

        # plot story
        t0 += 1

        lines[1].write(begin_time=t0)
        s_e.appear(begin_time=t0)
        s_e.write_name_as_label(modus='down', begin_time=t0 + 0.5)
        t0 += 2

        lines[0].move_copy_to_and_remove(lines[2], src_letter_indices=range(27, 47), target_letter_indices=range(0, 20),
                                         begin_time=t0, new_color='text', offset=[0, 0, -0.001])
        t0 += 1
        lines[2].write(letter_range=[20, 22], begin_time=t0, transition_time=0.1)
        t0 += 1

        lines[2].rescale(rescale=[1.4, 1.4, 1], begin_time=t0)
        t0 += 2

        lines[2].move_copy_to_and_remove(lines[3],
                                         src_letter_indices=[0, 1, 2, 3, 4, 5, 8, 9, 12, 15, 16, 17, 18, 19],
                                         target_letter_indices=range(0, 14), begin_time=t0, offset=[0, -0.26, -0.001])
        t0 += 1
        lines[3].write(letter_range=[14, 16], begin_time=t0, transition_time=0.2)
        t0 += 1

        # change sign and move terms to the other side
        lines[3].replace(lines[4], src_letter_range=[8, 14], img_letter_range=[8, 14], begin_time=t0)
        lines[3].move_letters_to(lines[5], src_letter_indices=range(8, 14), target_letter_indices=range(15, 21),
                                 begin_time=t0, offsets=[[0, 0, -0.001]])
        lines[3].letters[15].disappear(begin_time=t0, transition_time=0.5)
        t0 += 1
        lines[5].write(letter_range=[0, 8], begin_time=t0, transition_time=0)
        lines[5].write(letter_range=[14, 21], begin_time=t0, transition_time=0)
        lines[3].disappear(begin_time=t0)
        t0 += 1

        # isolate xi
        lines[5].replace2(lines[6], src_letter_range=[0, 8], img_letter_range=[0, 9], begin_time=t0)
        t0 += 1
        lines[6].write(letter_range=[15, 22], begin_time=t0, transition_time=0)
        lines[5].disappear(begin_time=t0, transition_time=0)
        t0 += 1

        # solve for xi
        # move numerator
        lines[6].move_letters_to(lines[7], src_letter_indices=[16, 17, 18, 19, 20, 21], offsets=[[0, 0.26, -0.001]],
                                 target_letter_indices=[17, 19, 21, 23, 26, 28], begin_time=t0, transition_time=0.5)
        t0 += 0.5
        lines[7].write(letter_set=[17, 19, 21, 23, 24, 26, 28], begin_time=t0,
                       transition_time=0)  # 24 is the fraction line
        set = [16, 17, 18, 19, 20, 21]
        for l in set:
            lines[6].letters[l].disappear(begin_time=t0, transition_time=0)

        # move denominator
        offsets = [[0, -0.18, -0.001], [0, -0.18, -0.001], [0, -0.18, -0.001], [0, -0.18, -0.001], [0, -0.22, -0.001],
                   [0, -0.18, -0.001]]
        lines[6].move_letters_to(lines[7], src_letter_indices=[1, 2, 3, 4, 5, 6, 7, 8],
                                 target_letter_indices=[16, 18, 20, 22, 25, 27, 29, 30], offsets=offsets
                                 , begin_time=t0, transition_time=1)
        t0 += 1
        lines[7].write(letter_set=[0, 15, 16, 18, 20, 22, 25, 27, 29, 30], begin_time=t0, transition_time=0)
        lines[6].disappear(begin_time=t0, transition_time=0)
        t0 += 1

        # move xi
        lines[7].move_letters_to(src_letter_indices=[0], target_letter_indices=[14], begin_time=t0)
        t0 += 1

        lines[8].write(begin_time=t0, transition_time=0)
        lines[8].move(direction=[-0.005, 0.0025, 0.001], begin_time=t0 - 0.1, transition_time=0)
        lines[9].write(begin_time=t0, transition_time=0)
        lines[9].letters[4].move(direction=[0, -0.035, 0.001], begin_time=t0 - 0.1, transition_time=0)

        set = [17, 19, 21, 23, 26, 28, 16, 18, 20, 22, 25, 27, 29, 30]
        for s in set:
            lines[7].letters[s].disappear(begin_time=t0, transition_time=0)

        lines[8].replace2(lines[11], begin_time=t0)
        t0 += 1.5
        lines[9].replace2(lines[12], begin_time=t0)
        t0 += 1.5

        dt = 1 / 8
        for i in range(1, 8):
            lines[11].letters[i].change_color(new_color='joker', begin_time=t0 + dt * i)
            lines[12].letters[i].change_color(new_color='joker', begin_time=t0 + dt * i)

        t0 += 2
        dt = 1 / 8
        lines[10].write(letter_set=[2], begin_time=t0, transition_time=0)
        t0 += 0.1
        for i in range(1, 8):
            lines[11].letters[i].disappear(begin_time=t0 + dt * (7 - i))
            lines[12].letters[i].disappear(begin_time=t0 + dt * (7 - i))
        lines[7].letters[24].disappear(begin_time=t0)
        t0 += 2.5

        # insert xi
        lines[13].write(begin_time=t0, letter_set=[0, 1], transition_time=0.2)
        t0 += 1
        lines[0].move_copy_to_and_remove(lines[13],
                                         src_letter_indices=[4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                             22, 23, 24],
                                         target_letter_indices=[2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                                23, 24, 25, 26],
                                         begin_time=t0, offset=[0, -0.22, -0.001], new_color='text')
        t0 += 1.5

        lines[11].move_copy_to_and_remove(lines[13], src_letter_indices=[0], target_letter_indices=[4], begin_time=t0,
                                          offset=[-0.347, 0.04, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[12].move_copy_to_and_remove(lines[13], src_letter_indices=[0], target_letter_indices=[6], begin_time=t0,
                                          offset=[-0.347, -0.14, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[10].move_copy_to_and_remove(lines[13], src_letter_indices=[2], target_letter_indices=[5], begin_time=t0,
                                          offset=[-0.347, -0.06, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        t0 += 1
        lines[11].move_copy_to_and_remove(lines[13], src_letter_indices=[0], target_letter_indices=[11], begin_time=t0,
                                          offset=[-0.622, 0.04, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[12].move_copy_to_and_remove(lines[13], src_letter_indices=[0], target_letter_indices=[13], begin_time=t0,
                                          offset=[-0.622, -0.14, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[10].move_copy_to_and_remove(lines[13], src_letter_indices=[2], target_letter_indices=[12], begin_time=t0,
                                          offset=[-0.622, -0.06, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')

        t0 += 1.5

        lines[0].move_copy_to_and_remove(lines[13],
                                         src_letter_indices=[25, 26, 33, 34, 37, 38, 40, 41, 47, 48],
                                         target_letter_indices=[27, 28, 37, 38, 43, 44, 46, 47, 53, 54],
                                         begin_time=t0, offset=[0, -0.22, -0.001], new_color='text')
        # numerator seperate
        lines[0].move_copy_to_and_remove(lines[14],
                                         src_letter_indices=[27, 28, 30, 31, 32, 36, 39, 42, 43, 44, 45, 46],
                                         target_letter_indices=[0, 1, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17],
                                         begin_time=t0, offset=[0, -0.22, -0.001], new_color='text')

        t0 += 1.5

        lines[11].move_copy_to_and_remove(lines[14], src_letter_indices=[0], target_letter_indices=[2], begin_time=t0,
                                          offset=[-1.552, 0.1, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[12].move_copy_to_and_remove(lines[14], src_letter_indices=[0], target_letter_indices=[4], begin_time=t0,
                                          offset=[-1.552, -0.08, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[10].move_copy_to_and_remove(lines[14], src_letter_indices=[2], target_letter_indices=[3], begin_time=t0,
                                          offset=[-1.552, 0, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[11].move_copy_to_and_remove(lines[14], src_letter_indices=[0], target_letter_indices=[8], begin_time=t0,
                                          offset=[-1.812, 0.1, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[12].move_copy_to_and_remove(lines[14], src_letter_indices=[0], target_letter_indices=[10], begin_time=t0,
                                          offset=[-1.812, -0.08, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        lines[10].move_copy_to_and_remove(lines[14], src_letter_indices=[2], target_letter_indices=[9], begin_time=t0,
                                          offset=[-1.812, 0, 0.001], rescale=[0.5, 0.5, 0.5], new_color='important')
        t0 += 1.5

        # simplify imaginary part
        lines[14].replace(lines[15], src_letter_range=[0, 12], img_letter_range=[0, 6], begin_time=t0)
        t0 += 1.5

        # cancel terms
        set = [3, 5, 6, 7, 12, 13, 14, 15]
        dt = 1 / 10
        count = 0
        for s in set:
            if s < 10:
                k = count
            else:
                k = 13 - count
            lines[14].letters[s].change_color(new_color='important', begin_time=t0 + k * dt)
            count += 1

        t0 += 1.5
        dt = 1 / 10
        count = 0
        for i in range(0, int(len(set) / 2)):
            lines[14].letters[set[i]].disappear(begin_time=t0 + count * dt)
            lines[14].letters[set[len(set) - 1 - i]].disappear(begin_time=t0 + count * dt)
            count += 1
        t0 += 1.5

        set = [8, 11, 16, 17]
        for s in set:
            lines[14].letters[s].change_color(new_color='joker', begin_time=t0)
            lines[14].letters[s].disappear(begin_time=t0 + 1)

        t0 += 2.5
        set = [27, 28, 37, 38, 43, 44, 46, 47, 53, 54]
        dt = 0.1
        k = 0
        for s in set:
            lines[13].letters[s].disappear(begin_time=t0 + k * dt)
            k += 1
        t0 += 2

        # write real part again

        # lines[16].write(begin_time=t0,transition_time=0.3)
        lines[16].write(letter_set=[0, 1, 7, 9, 11, 12, 14, 17], begin_time=t0, transition_time=0.3)
        t0 += 0.5
        lines[17].write(begin_time=t0)
        lines[18].write(begin_time=t0 + 1, transition_time=0.1)
        lines[19].write(begin_time=t0 + 1)
        t0 += 3
        lines[17].replace(lines[20], begin_time=t0)
        t0 += 1.5
        lines[19].replace(lines[21], begin_time=t0)
        t0 += 1.5

        # move transformed expressions to final expression
        lines[17].move_letters_to(lines[16], src_letter_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                                  target_letter_indices=[6, 8, 15, 16, 18, 19, 21, 22], begin_time=t0)
        lines[17].move_null_curves_to(lines[16], null_indices=[0], target_letter_indices=[20], begin_time=t0)

        lines[19].move_letters_to(lines[16], src_letter_indices=[0, 1, 2, 3, 4],
                                  target_letter_indices=[3, 4, 15, 16, 18], begin_time=t0)
        lines[19].move_null_curves_to(lines[16], null_indices=[0, 1, 2, 3], target_letter_indices=[19, 20, 21, 22],
                                      begin_time=t0)

        lines[16].write(letter_set=[2, 10, 13], begin_time=t0 + 0.5, transition_time=0.5)
        lines[18].move_letters_to(lines[16], src_letter_indices=[0],
                                  target_letter_indices=[5], begin_time=t0)

        t0 += 1.5

        set1 = [2, 3, 4, 5, 6, 8, 10]
        set2 = [7, 9, 12, 14, 17]

        #  stupid unfortunately
        dt = 0.5 / 6
        lines[16].letters[set1[0]].change_color(new_color='joker', begin_time=t0 - dt)
        for i in range(0, len(set2)):
            index = set1[i + 1]
            if index in [5]:
                lines[18].letters[0].change_color(new_color='joker', begin_time=t0 + i * dt)
            elif index in [3, 4]:
                lines[19].letters[index - 3].change_color(new_color='joker', begin_time=t0 + i * dt)
            elif index in [6, 8]:
                lines[17].letters[int(index / 7)].change_color(new_color='joker', begin_time=t0 + i * dt)
            else:
                lines[16].letters[set1[i + 1]].change_color(new_color='joker', begin_time=t0 + i * dt)
            lines[16].letters[set2[i]].change_color(new_color='joker', begin_time=t0 + i * dt)
        lines[16].letters[set1[6]].change_color(new_color='joker', begin_time=t0 + 0.5)

        t0 += 1.5

        dt = 0.5 / 6
        lines[16].letters[set1[0]].disappear(begin_time=t0 - dt)
        for i in range(0, len(set2)):
            index = set1[i + 1]
            if index in [5]:
                lines[18].letters[0].disappear(begin_time=t0 + i * dt)
            elif index in [3, 4]:
                lines[19].letters[index - 3].disappear(begin_time=t0 + i * dt)
            elif index in [6, 8]:
                lines[17].letters[int(index / 7)].disappear(begin_time=t0 + i * dt)
            else:
                lines[16].letters[set1[i + 1]].disappear(begin_time=t0 + i * dt)
            lines[16].letters[set2[i]].disappear(begin_time=t0 + i * dt)
        lines[16].letters[set1[6]].disappear(begin_time=t0 + 0.5)

        lines[16].letters[11].disappear(begin_time=t0 + 0.5)
        lines[16].letters[13].disappear(begin_time=t0 + 0.5)
        t0 += 1.5

        set = [2, 7]
        for s in set:
            lines[17].letters[s].disappear(begin_time=t0)
        lines[19].disappear(begin_time=t0, transition_time=0)

        set = [3, 4, 5, 6]
        letters = []
        for s in set:
            letters.append(lines[17].letters[s])
        letters.append(lines[17].created_null_curves[0])

        for l in letters:
            l.change_color(new_color='important', begin_time=t0)
            l.move(direction=[-1.5, -0.25, 0], begin_time=t0)

        s_e.appear(begin_time=t0)
        s_e.write_name_as_label(begin_time=t0, modus='down', name='E')
        lines[23].write(begin_time=t0+1)
        t0 += 2
        lines[24].write(letter_set=[0, 1, 3, 2], begin_time=t0)
        lines[22].write(begin_time=t0+1,transition_time=0.5)
        t0 += 2
        print(len(lines))
        display.turn(begin_time=t0, flat=True)
        t0 += 1
        title = SimpleTexBObject(r'\text{Further Reading}', aligned='center')
        display.set_title_back(title)
        title.write(begin_time=t0)
        t0 += 2
        coords.add_objects([l_de_long])

        print("last frame ", t0 * FRAME_RATE)

    def further_reading(self):
        cues = self.sub_scenes['further_reading']
        t0 = 0  # cues['start']

        ibpy.set_camera_location(location=[6, -20, 0])
        empty = EmptyCube(location=[6, 0, 0], name='Camera_Empty')
        empty.appear(begin_time=0, transition_time=0)
        ibpy.set_camera_view_to(target=empty)

        # meet with previous slide
        title = SimpleTexBObject(r'\text{Further Reading}', aligned='center')
        [a, b, c, t] = [0, 1, 0.75 + 1.5j, 0.75]

        d_title = 0
        d_problem = 0
        d_part1 = 0
        d_part2 = 0
        dic_old = self.construction(t0, title,
                                    locs=[a, b, c, t],
                                    durations=[d_title,
                                               d_problem,
                                               d_part1,
                                               d_part2], text=False, prefix="pre_")

        t0 += 0.1
        # retrieve variables
        coords_old = dic_old['coords']
        sphere_strings = ['s_a', 's_b', 's_c', 's_d', 's_e', 's_t']
        segment_strings = ['l_ab', 'l_ac', 'l_bc', 'l_bd', 'l_de', 'l_tc']
        arc_strings = ['cbd', 'bac','ra','ra2']

        spheres_old = [s_a, s_b, s_c, s_d, s_e, s_t] = get_from_dictionary(dic_old, sphere_strings)
        segments_old = [l_ab, l_ac, l_bc, l_bd, l_de, l_tc] = get_from_dictionary(dic_old, segment_strings)
        [cbd, bac,ra,ra2] = get_from_dictionary(dic_old, arc_strings)

        a_loc = coords_old.coords2location(z2p(a))
        b_loc = coords_old.coords2location(z2p(b))
        t_loc = coords_old.coords2location(z2p(t))
        c_loc = coords_old.coords2location(z2p(c))
        [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)

        d_zoom = 0
        # zoom in
        coords_old.zoom(zoom=2, begin_time=t0, transition_time=d_zoom)
        coords_old.move(direction=[-4, 0, -2], begin_time=t0, transition_time=d_zoom)
        for sphere in spheres_old:
            sphere.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0, transition_time=d_zoom)
        for seg in segments_old:
            seg.rescale(rescale=[0.5, 0.5, 1], begin_time=t0, transition_time=d_zoom)


        t0 += 0.1

        # prepare transition
        s_d.write_name_as_label(modus='up', name='D', begin_time=t0 + d_part1 / 4, transition_time=0)
        s_a.write_name_as_label(modus='down_right', name='A=0', begin_time=t0, transition_time=0)
        s_b.write_name_as_label(modus='down_right', name='B=1', begin_time=t0 + 0.3, transition_time=0)
        s_c.write_name_as_label(modus='up', name='C=x+yi', begin_time=t0 + 0.3, transition_time=0)

        l_tc.disappear(begin_time=t0, transition_time=0)
        l_bd.disappear(begin_time=t0, transition_time=0)
        l_bc.disappear(begin_time=t0, transition_time=0)
        l_de.disappear(transition_time=0)
        s_t.disappear(begin_time=t0, transition_time=0)
        bac.disappear(begin_time=t0, transition_time=0)
        cbd.disappear(begin_time=t0, transition_time=0)
        ra.disappear(begin_time=t0, transition_time=0)
        ra2.disappear(begin_time=t0, transition_time=0)
        s_e.disappear(transition_time=0)

        da = a_loc - d_loc
        ap_loc = Vector([-da.z, 0, da.x]) - da
        v = ap_loc - d_loc

        l_de_long = Cylinder.from_start_to_end(start=d_loc - 2 * v, end=ap_loc + 3 * v,
                                               color='example', thickness=0.26, name='l_de')
        l_de_long.grow(modus='from_start', begin_time=t0, transition_time=0)

        coords_old.add_object(l_de_long)
        t0 += 1.5

        coords_old.disappear(begin_time=t0)

        # the construction is outsourced
        d_title = 0
        d_problem = 0
        d_part1 = 1
        d_part2 = 1
        dic = self.construction(t0, title,
                                locs=[a, b, c, t],
                                durations=[d_title,
                                           d_problem,
                                           d_part1,
                                           d_part2], text=False)

        t0 += 0.1
        # retrieve variables
        coords = dic['coords']
        sphere_strings = ['s_a', 's_b', 's_c', 's_d', 's_e', 's_t']
        segment_strings = ['l_ab', 'l_ac', 'l_bc', 'l_bd', 'l_de', 'l_tc']
        arc_strings = ['cbd', 'bac','ra','ra2']

        spheres = [s_a, s_b, s_c, s_d, s_e, s_t] = get_from_dictionary(dic, sphere_strings)
        segments = [l_ab, l_ac, l_bc, l_bd, l_de, l_tc] = get_from_dictionary(dic, segment_strings)
        [cbd, bac,ra,ra2] = get_from_dictionary(dic, arc_strings)

        a_loc = coords.coords2location(z2p(a))
        b_loc = coords.coords2location(z2p(b))
        t_loc = coords.coords2location(z2p(t))
        c_loc = coords.coords2location(z2p(c))
        [d_loc, alpha, beta, e_loc] = geometry(a_loc, b_loc, c_loc, t_loc)

        # text work
        display = Display(scales=[4, 5], location=[12.5, 0, 0], number_of_lines=18, flat=True)
        display.set_title(title)
        title.write(transition_time=0)

        t0 += 2

        # zoom in
        d_zoom = 0
        coords.zoom(zoom=2, begin_time=t0, transition_time=d_zoom)
        coords.move(direction=[-4, 0, -2], begin_time=t0, transition_time=d_zoom)
        for sphere in spheres:
            sphere.rescale(rescale=[0.5, 0.5, 0.5], begin_time=t0, transition_time=d_zoom)
        for seg in segments:
            seg.rescale(rescale=[0.5, 0.5, 1], begin_time=t0, transition_time=d_zoom)
        t0 += 0.1

        # make dynamic again
        d_anim = 5
        direction = to_vector([0, 0, -3])
        s_c.move(direction=direction, begin_time=t0, transition_time=d_anim)
        steps = 10
        dt = d_anim / steps
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = geometry(a_loc, b_loc, c_new, t_loc)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bac.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            l_bc.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            l_bd.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            ra2.rotate(rotation_euler=[0, np.pi - alpha, 0], begin_time=t, transition_time=dt)
            ra2.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += d_anim

        s_c.move(direction=-direction, begin_time=t0, transition_time=d_anim)
        for s in range(0, steps):
            t = t0 + dt * s
            c_new = ibpy.get_location_at_frame(s_c, (t + dt) * FRAME_RATE)
            l_ac.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            [d_pos, alpha, beta, e_pos] = geometry(a_loc, b_loc, c_new, t_loc)
            l_de.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            s_d.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
            bac.extend_to(2 * alpha / np.pi, begin_time=t, transition_time=dt)
            cbd.extend_from_to(beta / np.pi, (beta + alpha) / np.pi, begin_time=t, transition_time=dt)
            l_bc.move_end_point(target_location=c_new, begin_time=t, transition_time=dt)
            l_bd.move_end_point(target_location=d_pos, begin_time=t, transition_time=dt)
            ra2.rotate(rotation_euler=[0, np.pi - alpha, 0], begin_time=t, transition_time=dt)
            ra2.move_to(target_location=d_pos, begin_time=t, transition_time=dt)
        t0 += d_anim

        t0 += 2
        print("last frame ", t0 * FRAME_RATE)

    def play(self, name):
        super().play()
        if hasattr(self, name):
            getattr(self, name)()


if __name__ == '__main__':
    try:
        example = BundesWettbewerbGeometry()
        dict = {}
        for i, scene in enumerate(example.sub_scenes):
            print(i, scene)
            dict[i] = scene
        choice = input("Choose scene:")
        print("Your choice: ", choice)
        selected_scene = dict[int(choice)]
        # selected_scene = 'basics2'
        example.create(name=selected_scene)
        # example.render(debug=True)
    except:
        print_time_report()
        raise ()
