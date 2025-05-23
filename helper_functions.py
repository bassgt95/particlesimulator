from manim import *
import numpy as np
from particle_simulator import *

def gaussian(v,m,T):
            a = m/(2*k_B*T)
            return np.sqrt(a/PI)*np.exp(-a*v**2)

def maxwell_boltzmann(v,m,T):
    a = np.sqrt(k_B*T/m)
    return np.sqrt(2/PI)*((v**2)/(a**3))*np.exp(-(v**2/(2*a**2)))

def MB_energy(E,T):
    return 2*np.sqrt(E/PI)*((1/(k_B*T))**(3/2))*np.exp(-E/(k_B*T))

def get_frequency_plotter(particles,getter,bucket_size):
    frequencies={}
    for particle in particles:
        value = 0
        match getter:
            case "get_speed": value = particle.get_speed()
            case "get_x_velocity": value = particle.get_x_velocity()
            case "get_y_velocity": value = particle.get_y_velocity()
            case "get_z_velocity": value = particle.get_z_velocity()
            case "get_kinetic_energy": value = particle.get_kinetic_energy()
        bucket=int(np.floor(value/bucket_size))
        if bucket in frequencies:
            frequencies[bucket]+=1
        else:
            frequencies[bucket]=1
    def plotter(x):
        bucket=int(np.floor(x/bucket_size))
        if bucket in frequencies:
            return frequencies[bucket]
        else:
            return 0
    return plotter

def magnitude(vector):
    return np.sqrt(np.dot(vector,vector))

def distance(position_1,position_2):
    return magnitude(np.subtract(position_1,position_2))

def cylindrical_to_cartesian(vector):
    rho=vector[0]
    phi=vector[1]
    z=vector[2]
    return np.array([
        rho*np.cos(phi),
        rho*np.sin(phi),
        z
    ])

def spherical_to_cartesian(vector):
    r=vector[0]
    theta=vector[1]
    phi=vector[2]
    return np.array([
        r*np.sin(phi)*np.cos(theta),
        r*np.sin(phi)*np.sin(theta),
        r*np.cos(phi),
    ])

def get_camera_unit_vector():
    return spherical_to_cartesian([
        1,
        self.renderer.camera.get_theta(),
        self.renderer.camera.get_phi()
    ])

def get_camera_position_vector():
    return spherical_to_cartesian([
        self.renderer.camera.get_focal_distance(),
        self.renderer.camera.get_theta(),
        self.renderer.camera.get_phi()
    ])

def get_camera_r():
    return self.renderer.camera.get_focal_distance()

def get_camera_theta():
    return self.renderer.camera.get_theta()

def get_camera_phi():
    return self.renderer.camera.get_phi()

def get_apparent_radius(actual_radius, distance_from_camera):
    # return distance_from_camera*actual_radius*np.sqrt(1-(actual_radius/distance_from_camera)**2)/(distance_from_camera**2-actual_radius**2)
    return get_camera_r()*actual_radius/(np.sqrt(distance_from_camera**2-actual_radius**2))

def get_z_index_from_position(position):
    return get_camera_r()-distance(position,get_camera_position_vector())

def add_wall_clack(location,direction):
    clack=Dot(point=location,radius=0).set_z_index(get_z_index_from_position(location))
    clack.add(turn_animation_into_updater(ClackFlash(clack,direction=direction,color=WHITE,line_length=0.1,line_stroke_width=3,rate_func=linear,run_time=0.075),cycle=False))
    self.add(clack)
    if sfx_on: self.add_sound("sounds/click")

def add_particle_clack(location,direction):
    clack=Dot(point=location,radius=0).set_z_index(get_z_index_from_position(location))
    clack.add(turn_animation_into_updater(ClackFlash(clack,direction=direction,color=YELLOW,line_length=0.1,line_stroke_width=3,rate_func=linear,run_time=0.035),cycle=False))
    self.add(clack)
    if sfx_on: self.add_sound("sounds/clack")

def handle_particle_collision(particle1,particle2,distance_between_particles):
    unit_vector=np.add(particle2.get_center(),-particle1.get_center())/distance_between_particles
    point_of_contact=np.add(particle1.get_center(),particle1.radius*unit_vector/(particle1.radius+particle2.radius))
    if particle_clacks_on: add_particle_clack(point_of_contact,unit_vector)

    m_1 = particle1.mass
    m_2 = particle2.mass

    # The dot product of the velocity vector with the unit vector gives
    # the magnitude of the net velocity in the direction of the collision:

    net_v_1 = np.dot(particle1.velocity,unit_vector)*unit_vector
    net_v_2 = np.dot(particle2.velocity,unit_vector)*unit_vector

    # These net pre-collision velocity vectors enable us to solve for
    # the net post-collision velocity vectors as a 1-dimensional collision:

    net_u_1 = np.add((m_1-m_2)*net_v_1,2*m_2*net_v_2)/(m_1+m_2)
    net_u_2 = np.add((m_2-m_1)*net_v_2,2*m_1*net_v_1)/(m_2+m_1)

    # The total post-collision velocity is the total pre-collision velocity
    # plus the difference between the net pre- & post-collision velocities:

    particle1.accelerate(np.subtract(net_u_1,net_v_1))
    particle2.accelerate(np.subtract(net_u_2,net_v_2))

def detect_particle_collision(particle1,particle2):
    # Check if distance between particle centers is less than the sum of their radii:
    distance_between_particles=distance(particle1.get_center(),particle2.get_center())
    if (distance_between_particles<particle1.radius+particle2.radius):
        # Check if the distance between particle centers is DECREASING:
        if np.dot(np.subtract(particle2.get_center(),particle1.get_center()),np.subtract(particle2.velocity,particle1.velocity))<0:
            handle_particle_collision(particle1,particle2,distance_between_particles)

def detect_box_collision(particle,dt):
    center=particle.get_center()
    radius=particle.radius
    velocity=particle.velocity
    # Check if particle intersects RIGHT WALL:
    if center[0]>container_center[0]+half_container_width_x-radius:
        # Check if particle is moving RIGHTWARD:
        if velocity[0]>0:
            handle_container_collision(particle,half_container_width_x,velocity[0],-damping_factor*velocity[0],X_AXIS,dt)
    # Check if particle intersects FRONT WALL:
    elif center[1]>container_center[1]+half_container_width_y-radius:
        # Check if particle is moving FORWARD:
        if velocity[1]>0:
            handle_container_collision(particle,half_container_width_y,velocity[1],-damping_factor*velocity[1],Y_AXIS,dt)
    # Check if particle intersects CEILING:
    elif center[2]>container_center[2]+half_container_height-radius:
        # Check if particle is moving UPWARD:
        if velocity[2]>0:
            handle_container_collision(particle,half_container_height,velocity[2],-damping_factor*velocity[2],Z_AXIS,dt)
    # Check if particle intersects LEFT WALL:
    elif center[0]<container_center[0]+-half_container_width_x+radius:
        # Check if particle is moving LEFTWARD:
        if velocity[0]<0:
            handle_container_collision(particle,half_container_width_x,-velocity[0],damping_factor*velocity[0],-X_AXIS,dt)
    # Check if particle intersects BACK WALL:
    elif center[1]<container_center[1]+-half_container_width_y+radius:
        # Check if particle is moving BACKWARD:
        if velocity[1]<0:
            handle_container_collision(particle,half_container_width_y,-velocity[1],damping_factor*velocity[1],-Y_AXIS,dt)
    # Check if particle intersects FLOOR:
    elif center[2]<container_center[2]+-half_container_height+radius:
        # Check if particle is moving DOWNWARD:
        if velocity[2]<0:
            handle_container_collision(particle,half_container_height,-velocity[2],damping_factor*velocity[2],-Z_AXIS,dt)

def detect_cylinder_collision(particle,dt):
    horizontal_position=np.array([particle.get_x(),particle.get_y(),0])
    distance_from_z_axis=magnitude(horizontal_position)
    # Check if particle intersects CYLINDRICAL WALL:
    if (distance_from_z_axis>container_radius-particle.radius):
        unit_vector=horizontal_position/distance_from_z_axis
        net_velocity=np.dot(particle.velocity,unit_vector)
        # Check if particle is moving OUTWARD FROM Z-AXIS:
        if (net_velocity>0):
            handle_container_collision(particle,container_radius,net_velocity,-damping_factor*net_velocity,unit_vector,dt)
    # Check if particle intersects CEILING:
    elif particle.get_z()>container_center[2]+half_container_height-particle.radius:
        # Check if particle is moving UPWARD:
        if particle.velocity[2]>0:
            handle_container_collision(particle,container_center[2]+half_container_height,particle.velocity[2],-damping_factor*particle.velocity[2],Z_AXIS,dt)
    # Check if particle intersects FLOOR:
    elif particle.get_z()<container_center[2]-half_container_height+particle.radius:
        # Check if particle is moving DOWNWARD:
        if particle.velocity[2]<0:
            handle_container_collision(particle,container_center[2]+half_container_height,-particle.velocity[2],damping_factor*particle.velocity[2],-Z_AXIS,dt)    

def detect_globe_collision(particle,dt):
    distance_from_origin=magnitude(particle.get_center())
    # Check if particle intersects SPHERICAL SHELL
    if (distance_from_origin>container_radius-particle.radius):
        unit_vector=particle.get_center()/distance_from_origin
        net_velocity=np.dot(particle.velocity,unit_vector)
        # Check if particle is moving OUTWARD FROM ORIGIN
        if net_velocity>0:
            handle_container_collision(particle,container_radius,net_velocity,-damping_factor*net_velocity,unit_vector,dt)

def handle_container_collision(particle,boundary,v1,v2,unit_vector,dt):
    position_1=np.dot(particle.get_center(),unit_vector)
    position_0=position_1-v1*dt
    t_c=(boundary-particle.radius-position_0)/(position_1-position_0)
    particle.shift(-t_c*dt*particle.velocity)
    if wall_clacks_on: add_wall_clack(np.add(particle.get_center(),particle.radius*unit_vector),unit_vector)
    particle.accelerate((v2-v1)*unit_vector)
    particle.shift((1-t_c)*dt*particle.velocity)
    particle.detect_container_collision((1-t_c)*dt)

def check_particle_collisions(num_divs):
    grid=[]
    for i1 in range(num_divs):
        row=[]
        for i2 in range(num_divs):
            column=[]
            for i3 in range(num_divs):
                column.append([])
            row.append(column)
        grid.append(row)
    for particle in particles:
        x = num_divs*(particle.get_x()/container_breadth+0.5)
        y = num_divs*(particle.get_y()/container_breadth+0.5)
        z = num_divs*(particle.get_z()/container_breadth+0.5)
        x_int = np.maximum(0,np.minimum(int(x),num_divs-1))
        x_frac = x - x_int
        y_int = np.maximum(0,np.minimum(int(y),num_divs-1))
        y_frac = y-y_int
        z_int = np.maximum(0,np.minimum(int(z),num_divs-1))
        z_frac = z - z_int
        scaled_radius=0.5*particle.radius*num_divs/half_container_height
        particle.add_to_cell(grid[x_int][y_int][z_int])
        # Particle intersects left wall:
        if (x_frac<scaled_radius and particle.get_x()>particle.radius-half_container_height):
            particle.add_to_cell(grid[x_int-1][y_int][z_int])
            # Particle intersects back-left edge:
            if (np.sqrt(x_frac**2+y_frac**2)<scaled_radius and particle.get_y()>particle.radius-half_container_height):
                particle.add_to_cell(grid[x_int-1][y_int-1][z_int])
                # Particle intersects back-bottom-left corner:
                if (np.sqrt(x_frac**2+y_frac**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                    particle.add_to_cell(grid[x_int-1][y_int-1][z_int-1])
                # Particle intersects back-top-left corner:
                if (np.sqrt(x_frac**2+y_frac**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                    particle.add_to_cell(grid[x_int-1][y_int-1][z_int+1])
            # Particle intersects front-left edge:
            if (np.sqrt(x_frac**2+(1-y_frac)**2)<scaled_radius and particle.get_y()<half_container_height-particle.radius):
                particle.add_to_cell(grid[x_int-1][y_int+1][z_int])
                # Particle intersects front-bottom-left corner:
                if (np.sqrt(x_frac**2+(1-y_frac)**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                    particle.add_to_cell(grid[x_int-1][y_int+1][z_int-1])
                # Particle intersects front-top-left corner:
                if (np.sqrt(x_frac**2+(1-y_frac)**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                    particle.add_to_cell(grid[x_int-1][y_int+1][z_int+1])
            # Particle intersects bottom-left edge:
            if (np.sqrt(x_frac**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                particle.add_to_cell(grid[x_int-1][y_int][z_int-1])
            # Particle intersects top-left edge:
            if (np.sqrt(x_frac**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                particle.add_to_cell(grid[x_int-1][y_int][z_int+1])
        # Particle intersects right wall:
        if (x_frac>1-scaled_radius and particle.get_x()<half_container_height-particle.radius):
            particle.add_to_cell(grid[x_int+1][y_int][z_int])
            # Particle intersects back-right edge:
            if (np.sqrt((1-x_frac)**2+y_frac**2)<scaled_radius and particle.get_y()>particle.radius-half_container_height):
                particle.add_to_cell(grid[x_int+1][y_int-1][z_int])
                # Particle intersects back-bottom-right corner:
                if (np.sqrt((1-x_frac)**2+y_frac**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                    particle.add_to_cell(grid[x_int+1][y_int-1][z_int-1])
                # Particle intersects back-top-right corner:
                if (np.sqrt((1-x_frac)**2+y_frac**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                    particle.add_to_cell(grid[x_int+1][y_int-1][z_int+1])
            # Particle intersects front-right edge:
            if (np.sqrt((1-x_frac)**2+(1-y_frac)**2)<scaled_radius and particle.get_y()<half_container_height-particle.radius):
                particle.add_to_cell(grid[x_int+1][y_int+1][z_int])
                # Particle intersects front-bottom-right corner:
                if (np.sqrt((1-x_frac)**2+(1-y_frac)**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                    particle.add_to_cell(grid[x_int+1][y_int-1][z_int-1])
                # Particle intersects front-top-right corner:
                if (np.sqrt((1-x_frac)**2+(1-y_frac)**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                    particle.add_to_cell(grid[x_int+1][y_int+1][z_int+1])
            # Particle intersects bottom-right edge:
            if (np.sqrt((1-x_frac)**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                particle.add_to_cell(grid[x_int+1][y_int][z_int-1])
            # Particle intersects top-right edge:
            if (np.sqrt((1-x_frac)**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                particle.add_to_cell(grid[x_int+1][y_int][z_int+1])
        # Particle intersects back wall:
        if (y_frac<scaled_radius and particle.get_y()>particle.radius-half_container_height):
            particle.add_to_cell(grid[x_int][y_int-1][z_int])
            # Particle intersects bottom-back edge:
            if (np.sqrt(y_frac**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                particle.add_to_cell(grid[x_int][y_int-1][z_int-1])
            # Particle intersects top-back edge:
            if (np.sqrt(y_frac**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                particle.add_to_cell(grid[x_int][y_int-1][z_int+1])
        # Particle intersects front wall:
        if (y_frac>1-scaled_radius and particle.get_y()<half_container_height-particle.radius):
            particle.add_to_cell(grid[x_int][y_int+1][z_int])
            # Particle intersects bottom-front edge:
            if (np.sqrt((1-y_frac)**2+z_frac**2)<scaled_radius and particle.get_z()>particle.radius-half_container_height):
                particle.add_to_cell(grid[x_int][y_int+1][z_int-1])
            # Particle intersects top-front edge:
            if (np.sqrt((1-y_frac)**2+(1-z_frac)**2)<scaled_radius and particle.get_z()<half_container_height-particle.radius):
                particle.add_to_cell(grid[x_int][y_int+1][z_int+1])
        # particle intersects floor:
        if (z_frac<scaled_radius and particle.get_z()>particle.radius-half_container_height):
            particle.add_to_cell(grid[x_int][y_int][z_int-1])
        # particle intersects ceiling:
        if (z_frac>1-scaled_radius and particle.get_z()<half_container_height-particle.radius):
            particle.add_to_cell(grid[x_int][y_int][z_int+1])