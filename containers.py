from manim import *
import numpy as np
from particle_simulator import *

# Containers

class BoxContainer(VGroup):
    def __init__(self, dimensions, color, opacity, sheen):
        self.half_width_x=dimensions[0]/2
        self.half_width_y=dimensions[1]/2
        self.half_height=dimensions[2]/2
        self.shape="box"
        self.surfaces=[
            Wall([
                [-self.half_width_x,-self.half_width_y,-self.half_height],
                [-self.half_width_x,-self.half_width_y,self.half_height],
                [-self.half_width_x,self.half_width_y,self.half_height],
                [-self.half_width_x,self.half_width_y,-self.half_height]
            ]).add_updater(lambda w: w.set_z_index(-np.inf*np.sin(get_camera_phi())*np.cos(get_camera_theta()))),
            Wall([
                [self.half_width_x,-self.half_width_y,-self.half_height],
                [self.half_width_x,-self.half_width_y,self.half_height],
                [self.half_width_x,self.half_width_y,self.half_height],
                [self.half_width_x,self.half_width_y,-self.half_height]
            ]).add_updater(lambda w: w.set_z_index(np.inf*np.sin(get_camera_phi())*np.cos(get_camera_theta()))),
            Wall([
                [-self.half_width_x,-self.half_width_y,-self.half_height],
                [-self.half_width_x,-self.half_width_y,self.half_height],
                [self.half_width_x,-self.half_width_y,self.half_height],
                [self.half_width_x,-self.half_width_y,-self.half_height]
            ]).add_updater(lambda w: w.set_z_index(-np.inf*np.sin(get_camera_phi())*np.sin(get_camera_theta()))),
            Wall([
                [-self.half_width_x,self.half_width_y,-self.half_height],
                [-self.half_width_x,self.half_width_y,self.half_height],
                [self.half_width_x,self.half_width_y,self.half_height],
                [self.half_width_x,self.half_width_y,-self.half_height]
            ]).add_updater(lambda w: w.set_z_index(np.inf*np.sin(get_camera_phi())*np.sin(get_camera_theta()))),
            Wall([
                [-self.half_width_x,-self.half_width_y,-self.half_height],
                [-self.half_width_x,self.half_width_y,-self.half_height],
                [self.half_width_x,self.half_width_y,-self.half_height],
                [self.half_width_x,-self.half_width_y,-self.half_height]
            ]).add_updater(lambda w: w.set_z_index(-np.inf*np.cos(get_camera_phi()))),
            Wall([
                [-self.half_width_x,-self.half_width_y,self.half_height],
                [-self.half_width_x,self.half_width_y,self.half_height],
                [self.half_width_x,self.half_width_y,self.half_height],
                [self.half_width_x,-self.half_width_y,self.half_height]
            ]).add_updater(lambda w: w.set_z_index(np.inf*np.cos(get_camera_phi())))
        ]

class CubicContainer(BoxContainer):
    def __init__(self, semi_height, color, opacity, sheen):
        super(CubicContainer, self).__init__([semi_height, semi_height, semi_height], color, opacity, sheen)

class CylindricalContainer(VGroup):
    def __init__(self, semi_height, color, opacity, sheen):
        super(CylindricalContainer, self).__init__([semi_height, semi_height, semi_height], color, opacity, sheen)

class SphericalContainer(VGroup):
    def __init__(self, semi_height, color, opacity, sheen):
        super(SphericalContainer, self).__init__([semi_height, semi_height, semi_height], color, opacity, sheen)

# Container Surfaces:

class Wall(Polygon):
    def __init__(self, vertices, color, opacity, sheen):
        super(Wall,self).__init__(*vertices,fill_color=color,color=WHITE,stroke_width=1,fill_opacity=opacity,sheen_factor=sheen)

class Shell(ArcPolygonFromArcs):
    def __init__(self, arcs, radius, semi_height, orientation, color, opacity, sheen):
        super(Shell, self).__init__(*arcs,fill_color=color,fill_opacity=opacity,sheen_factor=sheen)
        self.radius=radius
        self.semi_height=semi_height
        self.orientation=orientation
        self.add_updater(lambda s: s.update_appearance())
    def update_appearance(self):
        if np.absolute((get_camera_r()*np.sin(get_camera_phi())))<container_radius:
            self.become(Shell([
                    Arc(angle=-TAU,arc_center=[0,0,self.semi_height],radius=self.radius),
                    Arc(angle=TAU,arc_center=[0,0,-self.semi_height],radius=self.radius)
                ],radius=self.radius,semi_height=self.semi_height,orientation=self.orientation
            ))
            self.set_z_index(-np.inf)
        else:
            [x,y,angle]=self.get_apparent_edge_coords()
            y=self.orientation*y
            self.become(Shell([
                ArcBetweenPoints(start=[-x,   y,   -self.semi_height], end=[-x,  y,    self.semi_height],     angle=0),
                ArcBetweenPoints(start=[-x,   y,   self.semi_height],  end=[x,   y,    self.semi_height],     angle=-angle,arc_center=[0,0,self.semi_height]),
                ArcBetweenPoints(start=[x,    y,   self.semi_height],  end=[x,   y,    -self.semi_height],    angle=0),
                ArcBetweenPoints(start=[x,    y,   -self.semi_height], end=[-x,  y,    -self.semi_height],    angle=angle,arc_center=[0,0,-self.semi_height])
            ],
            radius=self.radius,
            semi_height=self.semi_height,orientation=self.orientation))
            self.set_z_index(-self.orientation*np.inf)
        self.rotate_about_origin(angle=self.orientation*0.5*PI+get_camera_theta(),axis=Z_AXIS)
    def get_apparent_edge_coords(self):
        r=get_camera_r()
        phi=get_camera_phi()
        y_coord=-(self.radius**2)/(r*np.sin(phi))
        x_coord=np.sqrt(self.radius**2-y_coord**2)
        angle=PI+self.orientation*(2*np.arccos(y_coord/self.radius)-PI)
        return [x_coord,y_coord,angle]

class Lid(Circle):
    def __init__(self, z_coord, radius, color, opacity, sheen):
        super(Lid, self).__init__(radius=radius,fill_color=color,color=WHITE,stroke_width=1,fill_opacity=opacity,sheen_factor=sheen)
        self.shift(z_coord*Z_AXIS)

class Globe(Circle):
    def __init__(self, radius, color, opacity, sheen):
        super(Globe, self).__init__(radius=radius,fill_color=color,color=WHITE,stroke_width=1,fill_opacity=opacity,sheen_factor=sheen)
        self.apparent_radius=radius
    def update_apparent_radius(self):
        new_apparent_radius=get_apparent_radius(self.radius,get_camera_r())
        self.scale(new_apparent_radius/self.apparent_radius)
        self.set_apparent_radius(new_apparent_radius)
        return self
    def set_apparent_radius(self, apparent_radius):
        self.apparent_radius=apparent_radius
        return self