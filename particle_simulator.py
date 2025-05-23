from manim import *
from functools import reduce
import numpy as np
from collections.abc import Iterable
from AABB import *
# from particle import Particle
# from containers import BoxContainer, CubicContainer
# from helper_functions import *
config.disable_caching=True

rng=np.random.default_rng()

class ParticleSimulation(ThreeDScene):
    def construct(self):

        ##################################
        ### Initialization Parameters: ###
        ##################################

        # Constants:

        # The Dalton, or Unified Atomic Mass Unit, in kilograms:
        Da=1.66053906892*(10**(-27))

        # The Boltzmann Constant, in joules per kelvin:
        k_B=1.380649*(10**(-23))

        # Avogadro's Number:
        N_A=6.02214076*(10**(23))
        
        # Molar Gas Constant, in joules per kelvin:
        R_=8.31446261815324

        # The Boltzmann constant, the molar gas constant, & Avogadro's number
        # are related to one another by the identity R_=k_B*N_A

        # Particle Parameters:

        particle_radius=0.85
        particle_mass=4*Da
        particle_color=PURE_RED
        max_initial_particle_height=1
        num_particles=1
        
        large_particle_on=False
        large_particle_radius=1
        large_particle_mass=10*Da
        large_particle_color=GREEN

        # Motion Paramaters:

        temperature=0.005
        total_kinetic_energy=1.5*num_particles*k_B*temperature
        gravity_on=False
        falling_rate=12
        damping_factor=1
        vectors_on=False
        trails_on=False
        afterimages_on=False

        # Collision Parameters:

        collisions_on=True
        wall_clacks_on=True
        particle_clacks_on=True
        sfx_on=True

        # Container Parameters:

        container_center=ORIGIN
        half_container_height=2.75
        half_container_width_x=1
        half_container_width_y=6.75
        container_radius=2.5
        is_flat=False
        if is_flat:
            half_container_height=particle_radius
        is_cube=True
        if is_cube:
            half_container_width_x=half_container_height
            half_container_width_y=half_container_height
        shapes=["box","cylinder","sphere"]
        container_shape=shapes[0]
        # container_breadth=np.maximum(2*half_container_height,2*container_radius)
        # container_volume=
        match container_shape:
            case "box":
                container_volume=8*half_container_width_x*half_container_width_y*half_container_height
            case "cylinder":
                container_volume=TAU*(container_radius**2)*half_container_height
            case "sphere":
                container_volume=(2/3)*TAU*(container_radius**3)
        pressure=total_kinetic_energy/container_volume
        # num_divs=np.maximum(1,int(container_breadth/(6*particle_radius)))
        container_color=GREEN_E.darker()
        container_opacity=0
        container_sheen=-0.25
        
        piston_on=False
        
        newtons_cradle_on=False
        pool_table_on=True

        # Animation Parameters:

        camera_initial_theta=0.05*TAU
        camera_initial_phi=0.4*PI
        camera_rotation_rate=1
        camera_rotation_axis="theta"
        focal_distance=12
        display_chart=False
        duration=2
        
        clacks=VGroup()
        trails=VGroup()

        #########################
        ### Helper Functions: ###
        #########################
        
        def render_shaded_sphere_image(radius_px, center, light_dir, color=GREEN, ambient=0.2):
            size = radius_px * 2
            img = np.zeros((size, size, 4), dtype=np.uint8)
            base_rgb = color_to_rgb(color)

            for y in range(size):
                for x in range(size):
                    dx = (x - radius_px) / radius_px
                    dy = (y - radius_px) / radius_px
                    if dx*dx + dy*dy <= 0.95:
                        dz = np.sqrt(1 - dx*dx - dy*dy)
                        normal = np.array([dx, dy, dz])
                        intensity = np.dot(normal, light_dir)
                        intensity = np.clip(intensity + ambient, 0, 1)
                        r, g, b = (np.array(base_rgb) * intensity * 255).astype(np.uint8)
                        img[y, x] = [r, g, b, 255]  # RGBA
                    elif dx*dx + dy*dy <= 1:
                        img[y, x] = [0, 0, 0, 255]  # Black outline
                    else:
                        img[y, x] = [0, 0, 0, 0]  # Transparent background

            # return Image.fromarray(img)
            return img
        
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
            flash=ClackFlash(clack,direction=direction,color=WHITE,line_length=0.1,line_stroke_width=3,rate_func=rate_functions.rush_from,run_time=0.1)
            clack.add(turn_animation_into_updater(flash,cycle=False))
            clack.add_updater(lambda c: check_frame(clack,flash))
            clacks.add(clack)
            
        def check_frame(clack,flash):
            if flash.frame>5:
                clack.clear_updaters()
                # flash.clean_up_from_scene(self)
                flash.lines.set_opacity(0)
                clacks.remove(*clack.submobjects, flash.lines)
            else:
                flash.frame+=1

        def add_particle_clack(location,direction):
            clack=Dot(point=location,radius=0).set_z_index(get_z_index_from_position(location))
            flash=ClackFlash(clack,direction=direction,color=YELLOW,line_length=0.1,line_stroke_width=3,rate_func=rate_functions.rush_from,run_time=0.1)
            clack.add(turn_animation_into_updater(flash,cycle=False))
            clack.add_updater(lambda c: check_frame(clack,flash))
            clacks.add(clack)

        def handle_particle_collision(particle1,particle2):
            unit_vector=np.subtract(particle2.get_center(),particle1.get_center())/(distance(particle1.get_center(),particle2.get_center()))
            point_of_contact=np.add(particle1.get_center(),particle1.radius*unit_vector)
            if particle_clacks_on: add_particle_clack(point_of_contact,unit_vector)
            # self.add_fixed_orientation_mobjects(Dot(point_of_contact,color=PURE_RED,radius=0.025))
            # self.add(Arrow(start=particle1.get_center(),end=np.add(particle1.get_center(),particle1.radius*unit_vector),color=PURE_BLUE),Arrow(start=particle2.get_center(),end=np.subtract(particle2.get_center(),particle2.radius*unit_vector),color=PURE_GREEN))

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
                    
        def detect_continuous_particle_collision(particle1, particle2):
            start_time=max(particle1.time_of_last_collision,particle2.time_of_last_collision)
            starting_position_1 = np.add(particle1.get_center(),particle1.velocity*(start_time-particle1.time_of_last_collision))
            starting_position_2 = np.add(particle2.get_center(),particle2.velocity*(start_time-particle2.time_of_last_collision))
            alpha = np.subtract(starting_position_1,starting_position_2)
            beta = np.subtract(np.subtract(particle1.target_position,particle2.target_position),alpha)
            a = np.dot(beta,beta)
            b = np.dot(alpha,beta)
            c = np.dot(alpha,alpha)-(particle1.radius+particle2.radius)**2
            gamma = b**2-a*c
            if gamma>0:
                time_of_impact=start_time+(1-start_time)*(-b-np.sqrt(gamma))/a
                if (time_of_impact < 1) and (time_of_impact >= particle1.time_of_last_collision) and (time_of_impact >= particle2.time_of_last_collision):
                    return [True, time_of_impact]
                else: return [False]
            else: return [False]

        def detect_box_collision(particle,dt):
            center=particle.target_position
            radius=particle.radius
            velocity=particle.velocity
            collision=None
            # Check if particle intersects RIGHT WALL:
            if center[0]>container_center[0]+half_container_width_x-radius:
                # Check if particle is moving RIGHTWARD:
                if velocity[0]>0:
                    collision = get_container_collision(particle,half_container_width_x,velocity[0],-damping_factor*velocity[0],X_AXIS,dt)
            # Check if particle intersects FRONT WALL:
            if center[1]>container_center[1]+half_container_width_y-radius:
                # Check if particle is moving FORWARD:
                if velocity[1]>0:
                    new_collision = get_container_collision(particle,half_container_width_y,velocity[1],-damping_factor*velocity[1],Y_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
            # Check if particle intersects CEILING:
            if center[2]>container_center[2]+half_container_height-radius:
                # Check if particle is moving UPWARD:
                if velocity[2]>0:
                    new_collision = get_container_collision(particle,half_container_height,velocity[2],-damping_factor*velocity[2],Z_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
                    # return get_container_collision(particle,half_container_height,velocity[2],-damping_factor*velocity[2],Z_AXIS,dt)
            # Check if particle intersects LEFT WALL:
            if center[0]<container_center[0]+-half_container_width_x+radius:
                # Check if particle is moving LEFTWARD:
                if velocity[0]<0:
                    new_collision = get_container_collision(particle,half_container_width_x,-velocity[0],damping_factor*velocity[0],-X_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
                    # return get_container_collision(particle,half_container_width_x,-velocity[0],damping_factor*velocity[0],-X_AXIS,dt)
            # Check if particle intersects BACK WALL:
            if center[1]<container_center[1]+-half_container_width_y+radius:
                # Check if particle is moving BACKWARD:
                if velocity[1]<0:
                    new_collision = get_container_collision(particle,half_container_width_y,-velocity[1],damping_factor*velocity[1],-Y_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
                    # return get_container_collision(particle,half_container_width_y,-velocity[1],damping_factor*velocity[1],-Y_AXIS,dt)
            # Check if particle intersects FLOOR:
            if center[2]<container_center[2]+-half_container_height+radius:
                # Check if particle is moving DOWNWARD:
                if velocity[2]<0:
                    new_collision = get_container_collision(particle,half_container_height,-velocity[2],damping_factor*velocity[2],-Z_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
                    # return get_container_collision(particle,half_container_height,-velocity[2],damping_factor*velocity[2],-Z_AXIS,dt)
            return collision
        
        def detect_cylinder_collision(particle,dt):
            center=particle.target_position
            collision = None
            horizontal_position=[center[0],center[1],0] #np.array([particle.get_x(),particle.get_y(),0])
            distance_from_z_axis=magnitude(horizontal_position)
            # Check if particle intersects CYLINDRICAL WALL:
            if (distance_from_z_axis>container_radius-particle.radius):
                unit_vector=horizontal_position/distance_from_z_axis
                net_velocity=np.dot(particle.velocity,unit_vector)
                # Check if particle is moving OUTWARD FROM Z-AXIS:
                if (net_velocity>0):
                    new_collision = get_container_collision(particle,container_radius,net_velocity,-damping_factor*net_velocity,unit_vector,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
            if piston_on:
                # Check if particle intersects PISTON:
                if center[2]>piston.get_z()+piston.velocity[2]*dt:
                    new_collision = get_piston_collision(particle,piston,particle.velocity[2],-damping_factor*particle.velocity[2],Z_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
            # Check if particle intersects CEILING:
            elif center[2]>container_center[2]+half_container_height-particle.radius:
                # Check if particle is moving UPWARD:
                if particle.velocity[2]>0:
                    new_collision = get_container_collision(particle,container_center[2]+half_container_height,particle.velocity[2],-damping_factor*particle.velocity[2],Z_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
            # Check if particle intersects FLOOR:
            if center[2]<container_center[2]-half_container_height+particle.radius:
                # Check if particle is moving DOWNWARD:
                if particle.velocity[2]<0:
                    new_collision = get_container_collision(particle,container_center[2]+half_container_height,-particle.velocity[2],damping_factor*particle.velocity[2],-Z_AXIS,dt)
                    if (not collision) or (new_collision.time_of_impact < collision.time_of_impact):
                        collision = new_collision
            return collision
        
        def detect_globe_collision(particle,dt):
            distance_from_origin=magnitude(particle.target_position)
            # Check if particle intersects SPHERICAL SHELL
            if (distance_from_origin>container_radius-particle.radius):
                unit_vector=particle.get_center()/distance_from_origin
                net_velocity=np.dot(particle.velocity,unit_vector)
                # Check if particle is moving OUTWARD FROM ORIGIN
                if net_velocity>0:
                    return get_container_collision(particle,container_radius,net_velocity,-damping_factor*net_velocity,unit_vector,dt)
        
        def get_container_collision(particle,boundary,v1,v2,unit_vector,dt):
            position_1=np.dot(particle.target_position,unit_vector)
            position_0=position_1-v1*dt
            # position_0=np.dot(particle.get_center(),unit_vector)
            # position_1=position_0+v1*dt
            time_of_impact=(boundary-particle.radius-position_0)/(position_1-position_0)
            return SurfaceCollision(particle,unit_vector,v1,v2,time_of_impact=time_of_impact,dt=dt,scene=self,afterimages=afterimages)

        def get_piston_collision(particle,piston,v1,v2,unit_vector,dt):
            boundary=piston.get_z()
            v2=((particle.mass-piston.mass)*v1+2*piston.mass*piston.velocity[2])/(particle.mass+piston.mass)
            position_1=np.dot(particle.target_position,unit_vector)
            position_0=position_1-v1*dt
            # position_0=np.dot(particle.get_center(),unit_vector)
            # position_1=position_0+v1*dt
            time_of_impact=(boundary-particle.get_z()-particle.radius)/(particle.velocity[2]-piston.velocity[2])
            return PistonCollision(particle,unit_vector,v1,v2,time_of_impact=time_of_impact,dt=dt,scene=self,afterimages=afterimages)

        afterimages = VGroup()
        if afterimages_on:
            self.add(always_redraw(lambda: afterimages))
            
        shadows_on=True
        shadows=VGroup()
        if shadows_on:
            self.add(always_redraw(lambda: shadows))
        
        collision_tracker_on=False
        if collision_tracker_on:
            collision_list = VGroup()
            self.add(always_redraw(lambda: collision_list.arrange(-Z_AXIS).set_z_index(np.inf)))
        
        def check_AABB_collisions(aabb_tree, particles, dt):
            # afterimages.become(VGroup())
            collisions = []
            for particle in particles:
                particle.clear_afterimages()
                # particle.target_position=np.add(particle.get_center(),particle.velocity*dt)
                particle.update_target_position(dt)
                surface_collision = particle.detect_container_collision(dt)
                if surface_collision:
                    collisions.append(surface_collision)
            aabb_collision_pairs = aabb_tree.query_collision()
            for aabb_collision_pair in aabb_collision_pairs:
                [particle1,particle2]=aabb_collision_pair
                if any(x != y for x, y in zip(particle1.velocity, particle2.velocity)):
                    test_result = detect_continuous_particle_collision(*aabb_collision_pair)
                    if test_result[0]:
                        collisions.append(ParticleCollision(*aabb_collision_pair,time_of_impact=test_result[1],dt=dt,scene=self,afterimages=afterimages))
                    elif collision_tracker_on:
                        circle1 = Circle(radius=0.25,fill_color=particle1.color,stroke_width=0,fill_opacity=1)
                        circle2 = Circle(radius=0.25,fill_color=particle2.color,stroke_width=0,fill_opacity=1)
                        list_item=VGroup()
                        # list_item.add(Text("["),circle1,Text(","),circle2,Text(","),Text("MISS"),Text("]")).arrange(RIGHT).apply_matrix(np.linalg.inv(self.camera.generate_rotation_matrix()))
                        collision_list.add(list_item)
            collisions.sort(key=lambda collision: collision.time_of_impact, reverse=True)
            if collisions and collision_tracker_on:
                collision_list.become(VGroup())
            while collisions:
                collision = collisions.pop()
                if collision_tracker_on:
                    list_item = VGroup()
                    if len(collision.particles) == 2:
                        [particle1,particle2] = collision.particles
                        circle1 = Circle(radius=0.25,fill_color=particle1.color,stroke_width=0,fill_opacity=1)
                        circle2 = Circle(radius=0.25,fill_color=particle2.color,stroke_width=0,fill_opacity=1)
                        list_item.add(Text("["),circle1,Text(","),circle2,Text(","),MathTex(str(round(collision.time_of_impact,3))),Text("]"))
                    else:
                        circle1 = Circle(radius=0.25,fill_color=collision.particles[0].color,stroke_width=0,fill_opacity=1)
                        list_item.add(Text("["),circle1,Text(","),Square(side_length=0.5,color=WHITE,stroke_width=2),Text(","),MathTex(str(round(collision.time_of_impact,3))),Text("]"))
                    collision_list.add(list_item.arrange(RIGHT).apply_matrix(np.linalg.inv(self.camera.generate_rotation_matrix())).shift([-1,-4,0]))
                collision.resolve(aabb_tree,collisions)
            for particle in particles:
                if trails_on: trails.add(Trail(scene=self,start=particle.get_center(),end=particle.target_position,stroke_color=WHITE))
                if afterimages_on:
                    particle.update_afterimages()
                    afterimages.add(particle.afterimages)
                particle.move_to(particle.target_position)
                particle.time_of_last_collision = 0
                particle.update_appearance()
                # particle.target_position = np.add(particle.get_center(), particle.velocity*dt)
                if gravity_on:
                    particle.fall(dt)
                particle.update_target_position(dt)
                aabb_tree.remove(particle.aabb)
                particle.aabb=particle.get_aabb()
                aabb_tree.insert(particle.aabb)
            if piston_on:
                piston.move_to(piston.target_position)
                piston.push(dt)
                piston.time_of_last_collision=0
                piston.update_target_position(dt)
                
        ########################
        ### Mobject Classes: ###
        ########################

        # Particle:

        class Particle(ImageMobject):
            def __init__(self,radius,mass,color,energy,scene,spawn_point=None):
                # super(Particle,self).__init__(radius=radius,fill_color=color,fill_opacity=1,
                #                               sheen_factor=-0.5,sheen_direction=-Z_AXIS,
                #                               stroke_width=0,stroke_color=BLACK
                #                               )
                self.img=render_shaded_sphere_image(radius_px=int(130*radius), center=ORIGIN, light_dir=np.array([-2,-2,5]), color=color, ambient=0.2)
                super(Particle,self).__init__(self.img)
                self.prototype=self.copy()
                self.radius=radius
                if not spawn_point:
                    self.spawn_random_location()
                else:
                    self.move_to(spawn_point)
                self.apparent_radius=get_camera_r()*radius/(get_camera_r()-get_z_index_from_position(self.get_center()))
                self.mass=mass
                self.color=color
                self.scene=scene
                self.spawn_random_velocity(energy)
                self.target_position=self.get_center()
                self.time_of_last_collision=0
                self.aabb=self.get_aabb()
                self.afterimages=VGroup()
                # self.shadow=Shadow(radius=self.radius,center=np.subtract(self.get_center(),[0,0,self.radius]))
            def spawn_random_location(self):
                if (container_shape=="box"):
                    self.move_to([
                        rng.uniform(low=self.radius-half_container_width_x,high=half_container_width_x-self.radius),
                        rng.uniform(low=self.radius-half_container_width_y,high=half_container_width_y-self.radius),
                        rng.uniform(low=self.radius-half_container_height,high=(2*max_initial_particle_height-1)*(half_container_height-self.radius)),
                    ])
                elif (container_shape=="cylinder"):
                    self.move_to(cylindrical_to_cartesian([
                        np.sqrt(rng.uniform(low=0,high=1))*(container_radius-self.radius),
                        rng.uniform(low=0,high=TAU),
                        rng.uniform(low=self.radius-half_container_height,high=(2*max_initial_particle_height-1)*(half_container_height-self.radius)),
                    ]))
                elif (container_shape=="sphere"):
                    self.move_to(spherical_to_cartesian([
                        max_initial_particle_height*(container_radius-self.radius)*np.cbrt(rng.uniform(low=0,high=1)),
                        rng.uniform(low=0,high=TAU),
                        0.5*PI+np.arcsin(rng.uniform(low=-1,high=1))
                    ]))
            def spawn_random_velocity(self,energy):
                if is_flat:
                    self.set_velocity(cylindrical_to_cartesian([np.sqrt(2*energy/self.mass),rng.uniform(low=0,high=TAU),0]))
                else:
                    self.set_velocity(spherical_to_cartesian([np.sqrt(2*energy/self.mass),rng.uniform(low=0,high=TAU),0.5*PI+np.arcsin(rng.uniform(low=-1,high=1))]))
            def get_mass(self):
                return self.mass
            def get_x_velocity(self):
                return self.velocity[0]
            def get_y_velocity(self):
                return self.velocity[1]
            def get_z_velocity(self):
                return self.velocity[2]
            def get_speed(self):
                return magnitude(self.velocity)
            def get_momentum(self):
                return self.mass*self.velocity
            def get_kinetic_energy(self):
                return 0.5*self.mass*np.dot(self.velocity,self.velocity)
            def get_potential_energy(self):
                PE=0
                if gravity_on:
                    PE+=self.mass*falling_rate*(self.get_z()+half_container_height-self.radius)
                return PE
            def get_energy(self):
                return self.get_kinetic_energy() + self.get_potential_energy()
            def get_aabb(self):
                return AABB(
                    min(self.get_x()-self.radius, self.target_position[0]-self.radius),
                    min(self.get_y()-self.radius, self.target_position[1]-self.radius),
                    min(self.get_z()-self.radius, self.target_position[2]-self.radius),
                    max(self.get_x()+self.radius, self.target_position[0]+self.radius),
                    max(self.get_y()+self.radius, self.target_position[1]+self.radius),
                    max(self.get_z()+self.radius, self.target_position[2]+self.radius),
                    obj=self
                )
            def set_velocity(self,vector):
                self.velocity=vector
                return self
            def set_size(self,mass):#,radius=np.cbrt(particle_mass)*particle_radius):
                self.mass=mass
                radius=particle_radius*np.cbrt(mass/particle_mass)
                self.radius=radius
                self.update_appearance()
                return self
            def accelerate(self,vector):
                self.set_velocity(np.add(self.velocity, vector))
                return self
            def fall(self, dt):
                if (self.get_z()>self.radius-half_container_height): self.accelerate(-falling_rate*dt*Z_AXIS)
            def update_appearance(self):
                distance_from_camera=distance(self.get_center(),get_camera_position_vector())
                new_z_index=get_camera_r()-distance_from_camera
                new_apparent_radius=get_apparent_radius(self.radius,distance_from_camera)
                # self.scale(new_apparent_radius/self.apparent_radius)
                # self.become(Circle(radius=self.radius,fill_color=self.color,fill_opacity=1,stroke_width=0).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix())).set_sheen(-0.5,-Z_AXIS).move_to(self.get_center()))
                self.become(ImageMobject(render_shaded_sphere_image(radius_px=int(130*self.radius), center=ORIGIN, light_dir=np.dot(spherical_to_cartesian([10, get_camera_theta(), 0.4*PI]), self.scene.camera.generate_rotation_matrix()), color=self.color, ambient=0.2)).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix())).move_to(self.get_center()))
                self.apparent_radius=new_apparent_radius
                # self.set_sheen(-np.sin(get_camera_phi()),-Z_AXIS)
                self.set_z_index(new_z_index)
                # self.shadow.move_to(np.subtract(self.get_center(),[0,0,self.radius]))
                # self.shadow.radius=self.apparent_radius
                # self.shadow.become(Shadow(radius=self.radius,center=np.subtract(self.get_center(),[0,0,self.radius])))
                # self.add(Text(str(round(self.get_speed(),2)),color=WHITE))
                return self
            def update_target_position(self,dt):
                if gravity_on:
                    v1=self.velocity
                    v2=np.subtract(self.velocity,[0,0,-falling_rate*dt*(1-self.time_of_last_collision)])
                    self.target_position = np.add(self.get_center(),0.5*np.add(v1,v2)*dt*(1-self.time_of_last_collision))
                else:
                    self.target_position = np.add(self.get_center(),self.velocity*dt*(1-self.time_of_last_collision))
            def update_afterimages(self):
                self.afterimages.add(Circle(radius=self.radius,fill_color=self.color,fill_opacity=0.25,stroke_width=0).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix())).set_sheen(-0.5,-Z_AXIS).move_to(self.get_center()).set_z_index(-np.inf))
                self.afterimages.become(self.afterimages)
            def clear_afterimages(self):
                self.afterimages.become(VGroup())
            def detect_container_collision(self,dt):
                if (container_shape=="box"):
                    return detect_box_collision(self,dt)
                elif (container_shape=="cylinder"):
                    return detect_cylinder_collision(self,dt)
                elif (container_shape=="sphere"):
                    return detect_globe_collision(self,dt)
            def add_to_cell(self,cell):
                for particle in cell:
                    detect_particle_collision(self,particle)
                cell.append(self)
                
        class Shadow(Circle):
            def __init__(self,radius,center):
                super(Shadow,self).__init__(radius=radius,fill_opacity=0.65,fill_color=BLACK,stroke_width=0)
                self.radius=radius
                self.set_z_index(1-np.inf)
                self.move_to(center)
                
        class Trail(Line):
            def __init__(self,scene,**kwargs):
                super(Trail,self).__init__(**kwargs)
                self.scene=scene
                self.stroke_width=0.5
                self.set_z_index(np.inf)
                self.add_updater(lambda x: x.fade())
            def fade(self):
                self.stroke_width*=0.9
                if self.stroke_width<0.1:
                    self.set_stroke_width(0)
                    self.clear_updaters()
                    trails.remove(*self.submobjects)

        # Containers

        class BoxContainer(VGroup):
            def __init__(self, dimensions, color,):
                x=dimensions[0]/2
                y=dimensions[1]/2
                z=dimensions[2]/2
                self.surfaces=[
                    Wall([[-x,-y,-z],[-x,-y,z],[-x,y,z],[-x,y,-z]]).add_updater(lambda w: walls[0].set_z_index(-np.inf*np.sin(self.renderer.camera.get_phi())*np.cos(self.renderer.camera.get_theta()))),
                    Wall([[x,-y,-z],[x,-y,z],[x,y,z],[x,y,-z]]).add_updater(lambda w: walls[1].set_z_index(np.inf*np.sin(self.renderer.camera.get_phi())*np.cos(self.renderer.camera.get_theta()))),
                    Wall([[-x,-y,-z],[-x,-y,z],[x,-y,z],[x,-y,-z]]).add_updater(lambda w: walls[2].set_z_index(-np.inf*np.sin(self.renderer.camera.get_phi())*np.sin(self.renderer.camera.get_theta()))),
                    Wall([[-x,y,-z],[-x,y,z],[x,y,z],[x,y,-z]]).add_updater(lambda w: walls[3].set_z_index(np.inf*np.sin(self.renderer.camera.get_phi())*np.sin(self.renderer.camera.get_theta()))),
                    Wall([[-x,-y,-z],[-x,y,-z],[x,y,-z],[x,-y,-z]]).add_updater(lambda w: walls[4].set_z_index(-np.inf*np.cos(self.renderer.camera.get_phi()))),
                    Wall([[-x,-y,z],[-x,y,z],[x,y,z],[x,-y,z]]).add_updater(lambda w: walls[5].set_z_index(np.inf*np.cos(self.renderer.camera.get_phi())))
                ]

        # Container Surfaces:
        
        class Wall(Polygon):
            def __init__(self, vertices):
                super(Wall,self).__init__(*vertices,fill_color=container_color,color=WHITE,stroke_width=1,fill_opacity=container_opacity,sheen_factor=container_sheen)

        class Shell(ArcPolygonFromArcs):
            def __init__(self, arcs, radius, semi_height, orientation):
                super(Shell, self).__init__(*arcs,fill_color=container_color,fill_opacity=container_opacity,sheen_factor=container_sheen)
                self.radius=radius
                self.semi_height=semi_height
                self.orientation=orientation
                self.add_updater(lambda s: s.update_appearance())
            def update_appearance(self):
                if np.absolute((get_camera_r()*np.sin(get_camera_phi())))<container_radius:
                    self.become(Shell([
                            Arc(angle=-TAU,arc_center=[0,0,half_container_height],radius=self.radius),
                            Arc(angle=TAU,arc_center=[0,0,-half_container_height],radius=self.radius)
                        ],radius=self.radius,semi_height=self.semi_height,orientation=self.orientation
                    ))
                    self.set_z_index(-np.inf)
                else:
                    [x,y,angle]=self.get_apparent_edge_coords()
                    y=self.orientation*y
                    self.become(Shell([
                        ArcBetweenPoints(start=[-x,   y,   -half_container_height], end=[-x,  y,    half_container_height],     angle=0),
                        ArcBetweenPoints(start=[-x,   y,   half_container_height],  end=[x,   y,    half_container_height],     angle=-angle,arc_center=[0,0,half_container_height]),
                        ArcBetweenPoints(start=[x,    y,   half_container_height],  end=[x,   y,    -half_container_height],    angle=0),
                        ArcBetweenPoints(start=[x,    y,   -half_container_height], end=[-x,  y,    -half_container_height],    angle=angle,arc_center=[0,0,-half_container_height])
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
            def __init__(self, z_coord, radius):
                super(Lid, self).__init__(radius=radius,fill_color=container_color,color=WHITE,stroke_width=1,fill_opacity=container_opacity,sheen_factor=container_sheen)
                self.shift(z_coord*Z_AXIS)

        class Globe(Circle):
            def __init__(self, radius):
                super(Globe, self).__init__(radius=radius,fill_color=container_color,color=WHITE,stroke_width=1,fill_opacity=container_opacity,sheen_factor=container_sheen)
                self.apparent_radius=radius
            def update_apparent_radius(self):
                new_apparent_radius=get_apparent_radius(self.radius,get_camera_r())
                self.scale(new_apparent_radius/self.apparent_radius)
                self.set_apparent_radius(new_apparent_radius)
                return self
            def set_apparent_radius(self, apparent_radius):
                self.apparent_radius=apparent_radius
                return self

        class Piston(Circle):
            def __init__(self,mass,force,starting_height=half_container_height):
                super(Piston,self).__init__(radius=container_radius,stroke_width=0,fill_color=GRAY_D,fill_opacity=0.5,sheen_factor=1)
                # self.radius=container_radius
                self.mass=mass
                self.force=force
                self.acceleration=force/mass
                self.velocity=ORIGIN
                self.move_to(starting_height*Z_AXIS)
                self.target_position=self.get_center()
                self.time_of_last_collision=0
                # self.add_updater(lambda p, dt: p.shift(p.velocity*dt))
                # self.add_updater(lambda p, dt: p.push(dt))
                self.add_updater(lambda w: w.set_z_index(np.inf*np.dot(Z_AXIS*half_container_height,np.add(get_camera_position_vector(),-Z_AXIS*half_container_height))))
            def push(self,dt):
                self.velocity=np.subtract(self.velocity,[0,0,self.acceleration*dt])
                return self
            def detect_collision(self,particle):
                return self
            def update_target_position(self,dt):
                self.target_position=np.add(self.get_center(), self.velocity*dt*(1-self.time_of_last_collision))
                return self     
        
        # Collision Objects:
        
        class ParticleCollision:
            def __init__(self,particle1,particle2,time_of_impact,dt,scene,afterimages):
                self.particles=[particle1,particle2]
                self.time_of_impact=time_of_impact
                self.prev_collision_times=[particle1.time_of_last_collision,particle2.time_of_last_collision]
                self.dt=dt
                self.scene=scene
                self.afterimages=afterimages
            def resolve(self,aabb_tree,collisions):
                [particle1,particle2] = self.particles
                if (particle1.time_of_last_collision == self.prev_collision_times[0]) and (particle2.time_of_last_collision == self.prev_collision_times[1]):
                    new_position1=np.add(particle1.get_center(),self.dt*(self.time_of_impact-particle1.time_of_last_collision)*particle1.velocity)
                    if trails_on: trails.add(Trail(scene=self.scene,start=particle1.get_center(),end=new_position1,stroke_color=WHITE))
                    particle1.move_to(new_position1)
                    new_position2=np.add(particle2.get_center(),self.dt*(self.time_of_impact-particle2.time_of_last_collision)*particle2.velocity)
                    if trails_on: trails.add(Trail(scene=self.scene,start=particle2.get_center(),end=new_position2,stroke_color=WHITE))
                    particle2.move_to(new_position2)
                    handle_particle_collision(particle1, particle2)
                    if sfx_on: self.scene.add_sound(sound_file="sounds/clack",time_offset=(0)*self.dt)
                    particle1.time_of_last_collision=self.time_of_impact
                    particle2.time_of_last_collision=self.time_of_impact
                    # particle1.target_position = np.add(new_position1,particle1.velocity*self.dt*(1-particle1.time_of_last_collision))
                    # particle2.target_position = np.add(new_position2,particle2.velocity*self.dt*(1-particle2.time_of_last_collision))
                    particle1.update_target_position(self.dt)
                    particle2.update_target_position(self.dt)                    
                    for particle in self.particles:
                        particle.update_appearance()
                        if afterimages_on:
                            particle.update_afterimages()
                            # self.afterimages.add(Circle(radius=particle.apparent_radius,stroke_width=0,fill_color=particle.color,fill_opacity=self.time_of_impact).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix())).move_to(particle.get_center()))
                        aabb_tree.remove(particle.aabb)
                        particle.aabb=particle.get_aabb()
                        surface_collision = particle.detect_container_collision(self.dt)
                        if surface_collision:
                            # self.scene.add_fixed_position_mobjects(Square(side_length=2,color=PURE_BLUE))
                            collisions.append(surface_collision)
                        aabb_tree.insert(particle.aabb)
                    aabb_collision_pairs = aabb_tree.query_collision()
                    for aabb_collision_pair in aabb_collision_pairs:
                        [particle1,particle2]=aabb_collision_pair
                        if any(x != y for x, y in zip(particle1.velocity, particle2.velocity)):
                            # self.scene.add_fixed_in_frame_mobjects(Square(side_length=1.1,color=PURE_GREEN))
                            test_result = detect_continuous_particle_collision(*aabb_collision_pair)
                            if test_result[0]:
                                # self.scene.add_fixed_in_frame_mobjects(Square(side_length=1,color=PURE_RED))
                                collisions.append(ParticleCollision(*aabb_collision_pair,time_of_impact=test_result[1],dt=self.dt,scene=self.scene,afterimages=self.afterimages))
                            elif collision_tracker_on:
                                circle1 = Circle(radius=0.25,fill_color=particle1.color,stroke_width=0,fill_opacity=1)
                                circle2 = Circle(radius=0.25,fill_color=particle2.color,stroke_width=0,fill_opacity=1)
                                list_item=VGroup()
                                # list_item.add(Text("["),circle1,Text(","),circle2,Text(","),Text("MISS"),Text("]")).arrange(RIGHT).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix()))
                                collision_list.add(list_item)
                    collisions.sort(key=lambda collision: collision.time_of_impact, reverse=True)
                else: return
                
        class SurfaceCollision:
            def __init__(self,particle,unit_vector,v1,v2,time_of_impact,dt,scene,afterimages):
                self.particles=[particle]
                self.prev_collision_time=particle.time_of_last_collision
                self.unit_vector=unit_vector
                self.v1=v1
                self.v2=v2
                self.dt=dt
                self.time_of_impact=time_of_impact
                self.scene=scene
                self.afterimages=afterimages
            def resolve(self,aabb_tree,collisions):
                [particle]=self.particles
                if (particle.time_of_last_collision == self.prev_collision_time):
                    new_position=np.add(particle.get_center(),self.dt*(self.time_of_impact-particle.time_of_last_collision)*particle.velocity)
                    if trails_on: trails.add(Trail(scene=self.scene,start=particle.get_center(),end=new_position,stroke_color=WHITE))
                    particle.move_to(new_position)
                    if wall_clacks_on: add_wall_clack(np.add(particle.get_center(),particle.radius*self.unit_vector),self.unit_vector)
                    if sfx_on: self.scene.add_sound(sound_file="sounds/click",time_offset=(0)*self.dt)
                    particle.accelerate((self.v2-self.v1)*self.unit_vector)
                    particle.time_of_last_collision = self.time_of_impact
                    # particle.target_position = np.add(particle.get_center(),particle.velocity*self.dt*(1-particle.time_of_last_collision))
                    particle.update_target_position(self.dt)
                    for particle in self.particles:
                        particle.update_appearance()
                        if afterimages_on:
                            particle.update_afterimages()
                        #     self.afterimages.add(Circle(radius=particle.apparent_radius,stroke_width=0,fill_color=particle.color,fill_opacity=self.time_of_impact).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix())).move_to(particle.get_center()))
                        aabb_tree.remove(particle.aabb)
                        particle.aabb=particle.get_aabb()
                        surface_collision = particle.detect_container_collision(self.dt)
                        if surface_collision:
                            collisions.append(surface_collision)
                        aabb_tree.insert(particle.aabb)
                    aabb_collision_pairs = aabb_tree.query_collision()
                    for aabb_collision_pair in aabb_collision_pairs:
                        [particle1,particle2]=aabb_collision_pair
                        if any(x != y for x, y in zip(particle1.velocity, particle2.velocity)):
                            test_result = detect_continuous_particle_collision(*aabb_collision_pair)
                            if test_result[0]:
                                collisions.append(ParticleCollision(*aabb_collision_pair,test_result[1],self.dt,scene=self.scene,afterimages=self.afterimages))
                    collisions.sort(key=lambda collision: collision.time_of_impact, reverse=True)
                else: return
                
        class PistonCollision:
            def __init__(self,particle,unit_vector,v1,v2,time_of_impact,dt,scene,afterimages):
                self.particles=[particle]
                self.prev_collision_time=particle.time_of_last_collision
                self.unit_vector=unit_vector
                self.v1=v1
                self.v2=v2
                self.dt=dt
                self.time_of_impact=time_of_impact
                self.scene=scene
                self.afterimages=afterimages
            def resolve(self,aabb_tree,collisions):
                [particle]=self.particles
                if (particle.time_of_last_collision == self.prev_collision_time):
                    new_position=np.add(particle.get_center(),self.dt*(self.time_of_impact-particle.time_of_last_collision)*particle.velocity)
                    if trails_on: trails.add(Trail(scene=self.scene,start=particle.get_center(),end=new_position,stroke_color=WHITE))
                    particle.move_to(new_position)
                    piston.shift(piston.velocity*self.dt*self.time_of_impact)
                    if wall_clacks_on: add_wall_clack(np.add(particle.get_center(),particle.radius*self.unit_vector),self.unit_vector)
                    if sfx_on: self.scene.add_sound(sound_file="sounds/click",time_offset=(0)*self.dt)
                    piston.velocity=Z_AXIS*((piston.mass-particle.mass)*piston.velocity[2]+2*particle.mass*self.v1)/(piston.mass+particle.mass)
                    particle.accelerate((self.v2-self.v1)*self.unit_vector)
                    particle.time_of_last_collision = self.time_of_impact
                    piston.time_of_last_collision = self.time_of_impact
                    # particle.target_position = np.add(particle.get_center(),particle.velocity*self.dt*(1-particle.time_of_last_collision))
                    particle.update_target_position(self.dt)
                    for particle in self.particles:
                        particle.update_appearance()
                        if afterimages_on:
                            particle.update_afterimages()
                        #     self.afterimages.add(Circle(radius=particle.apparent_radius,stroke_width=0,fill_color=particle.color,fill_opacity=self.time_of_impact).apply_matrix(np.linalg.inv(self.scene.camera.generate_rotation_matrix())).move_to(particle.get_center()))
                        aabb_tree.remove(particle.aabb)
                        particle.aabb=particle.get_aabb()
                        surface_collision = particle.detect_container_collision(self.dt)
                        if surface_collision:
                            collisions.append(surface_collision)
                        aabb_tree.insert(particle.aabb)
                    aabb_collision_pairs = aabb_tree.query_collision()
                    for aabb_collision_pair in aabb_collision_pairs:
                        [particle1,particle2]=aabb_collision_pair
                        if any(x != y for x, y in zip(particle1.velocity, particle2.velocity)):
                            test_result = detect_continuous_particle_collision(*aabb_collision_pair)
                            if test_result[0]:
                                collisions.append(ParticleCollision(*aabb_collision_pair,test_result[1],self.dt,scene=self.scene,afterimages=self.afterimages))
                    collisions.sort(key=lambda collision: collision.time_of_impact, reverse=True)
                else: return

        class ClackFlash(AnimationGroup):

            def __init__(
                self,
                point: np.ndarray | Mobject,
                direction: np.ndarray,
                line_length: float = 0.2,
                num_lines: int = 12,
                flash_radius: float = 0.1,
                line_stroke_width: int = 3,
                color: str = YELLOW,
                time_width: float = 1,
                run_time: float = 1.0,
                **kwargs,
            ) -> None:
                if isinstance(point, Mobject):
                    self.point = point.get_center()
                else:
                    self.point = point
                self.frame=0
                self.color = color
                self.direction = direction
                self.line_length = line_length
                self.num_lines = num_lines
                self.flash_radius = flash_radius
                self.line_stroke_width = line_stroke_width
                self.run_time = run_time
                self.time_width = time_width
                self.animation_config = kwargs

                self.lines = self.create_lines()
                animations = self.create_line_anims()
                super().__init__(*animations, group=self.lines, opacity=0.5)

            def create_lines(self) -> VGroup:
                lines = VGroup()
                for angle in np.arange(0, TAU, TAU / self.num_lines):
                    line = Line(self.point, self.point + self.line_length * RIGHT)
                    line.shift((self.flash_radius) * RIGHT)
                    line.rotate(angle, about_point=self.point)
                    line.set_z_index(get_z_index_from_position(self.point+0.5*self.line_length))
                    line.set_opacity(0.5)
                    lines.add(line)
                lines.set_color(self.color)
                lines.set_stroke(width=self.line_stroke_width)
                lines.rotate(angle=angle_between_vectors(Z_AXIS,self.direction),axis=np.cross(Z_AXIS,self.direction))
                return lines

            def create_line_anims(self) -> Iterable[ShowPassingFlash]:
                return [
                    ShowPassingFlash(
                        line,
                        time_width=self.time_width,
                        run_time=self.run_time,
                        **self.animation_config,
                    )
                    for line in self.lines
                ]
            
        ########################
        ### Container Setup: ###
        ########################

        if (container_shape=="box"):
            walls=[
                Wall([[-half_container_width_x,-half_container_width_y,-half_container_height],[-half_container_width_x,-half_container_width_y,half_container_height],[-half_container_width_x,half_container_width_y,half_container_height],[-half_container_width_x,half_container_width_y,-half_container_height]]).add_updater(lambda w: w.set_z_index(np.inf*np.dot(-X_AXIS*half_container_width_x,np.add(get_camera_position_vector(),X_AXIS*half_container_width_x)))),
                Wall([[half_container_width_x,-half_container_width_y,-half_container_height],[half_container_width_x,-half_container_width_y,half_container_height],[half_container_width_x,half_container_width_y,half_container_height],[half_container_width_x,half_container_width_y,-half_container_height]]).add_updater(lambda w: w.set_z_index(np.inf*np.dot(X_AXIS*half_container_width_x,np.add(get_camera_position_vector(),-X_AXIS*half_container_width_x)))),
                Wall([[-half_container_width_x,-half_container_width_y,-half_container_height],[-half_container_width_x,-half_container_width_y,half_container_height],[half_container_width_x,-half_container_width_y,half_container_height],[half_container_width_x,-half_container_width_y,-half_container_height]]).add_updater(lambda w: w.set_z_index(np.inf*np.dot(-Y_AXIS*half_container_width_y,np.add(get_camera_position_vector(),Y_AXIS*half_container_width_y)))),
                Wall([[-half_container_width_x,half_container_width_y,-half_container_height],[-half_container_width_x,half_container_width_y,half_container_height],[half_container_width_x,half_container_width_y,half_container_height],[half_container_width_x,half_container_width_y,-half_container_height]]).add_updater(lambda w: w.set_z_index(np.inf*np.dot(Y_AXIS*half_container_width_y,np.add(get_camera_position_vector(),-Y_AXIS*half_container_width_y)))),
                Wall([[-half_container_width_x,-half_container_width_y,-half_container_height],[-half_container_width_x,half_container_width_y,-half_container_height],[half_container_width_x,half_container_width_y,-half_container_height],[half_container_width_x,-half_container_width_y,-half_container_height]]).add_updater(lambda w: w.set_z_index(np.inf*np.dot(-Z_AXIS*half_container_height,np.add(get_camera_position_vector(),Z_AXIS*half_container_height)))),
                Wall([[-half_container_width_x,-half_container_width_y,half_container_height],[-half_container_width_x,half_container_width_y,half_container_height],[half_container_width_x,half_container_width_y,half_container_height],[half_container_width_x,-half_container_width_y,half_container_height]]).add_updater(lambda w: w.set_z_index(np.inf*np.dot(Z_AXIS*half_container_height,np.add(get_camera_position_vector(),-Z_AXIS*half_container_height)))).set_opacity(0)
            ]
            if pool_table_on:
                # for wall in walls: wall.clear_updaters()
                walls+=[
                    Prism([2*half_container_width_x,0.2,2*half_container_height]).set_fill(color=GREEN_E.darker(),opacity=container_opacity).set_sheen(-0.25).shift([0,half_container_width_y+0.1,0]).add_updater(lambda w: w.set_z_index((np.inf+1)*np.dot(Y_AXIS*half_container_width_x,np.add(get_camera_position_vector(),Y_AXIS*half_container_width_y)))),
                    Prism([2*half_container_width_x,0.2,2*half_container_height]).set_fill(color=GREEN_E.darker(),opacity=container_opacity).set_sheen(-0.25).shift([0,-(half_container_width_y+0.1),0]).add_updater(lambda w: w.set_z_index((np.inf+1)*np.dot(-Y_AXIS*half_container_width_x,np.add(get_camera_position_vector(),Y_AXIS*half_container_width_y)))),
                    Prism([0.2,2*half_container_width_y+0.4,2*half_container_height]).set_fill(color=GREEN_E.darker(),opacity=container_opacity).set_sheen(-0.25).shift([half_container_width_x+0.1,0,0]).add_updater(lambda w: w.set_z_index((np.inf+1)*np.dot(X_AXIS*half_container_width_x,np.add(get_camera_position_vector(),X_AXIS*half_container_width_x)))),
                    Prism([0.2,2*half_container_width_y+0.4,2*half_container_height]).set_fill(color=GREEN_E.darker(),opacity=container_opacity).set_sheen(-0.25).shift([-(half_container_width_x+0.1),0,0]).add_updater(lambda w: w.set_z_index((np.inf+1)*np.dot(-X_AXIS*half_container_width_x,np.add(get_camera_position_vector(),X_AXIS*half_container_width_x)))),
                    # Wall([[-(half_container_width_x+0.2),-(half_container_width_y+0.2),-half_container_height],[-(half_container_width_x+0.2),-(half_container_width_y+0.2),half_container_height],[-(half_container_width_x+0.2),(half_container_width_y+0.2),half_container_height],[-(half_container_width_x+0.2),(half_container_width_y+0.2),-half_container_height]]).set_fill(color=DARK_BROWN,opacity=1).add_updater(lambda w: w.set_z_index(np.inf*np.dot(-X_AXIS*(half_container_width_x+0.2),np.add(get_camera_position_vector(),X_AXIS*(half_container_width_x+0.2))))),
                    # Wall([[(half_container_width_x+0.2),-(half_container_width_y+0.2),-half_container_height],[(half_container_width_x+0.2),-(half_container_width_y+0.2),half_container_height],[(half_container_width_x+0.2),(half_container_width_y+0.2),half_container_height],[(half_container_width_x+0.2),(half_container_width_y+0.2),-half_container_height]]).set_fill(color=DARK_BROWN,opacity=1).add_updater(lambda w: w.set_z_index(np.inf*np.dot(X_AXIS*(half_container_width_x+0.2),np.add(get_camera_position_vector(),-X_AXIS*(half_container_width_x+0.2))))),
                    # Wall([[-(half_container_width_x+0.2),-(half_container_width_y+0.2),-half_container_height],[-(half_container_width_x+0.2),-(half_container_width_y+0.2),half_container_height],[(half_container_width_x+0.2),-(half_container_width_y+0.2),half_container_height],[(half_container_width_x+0.2),-(half_container_width_y+0.2),-half_container_height]]).set_fill(color=DARK_BROWN,opacity=1).add_updater(lambda w: w.set_z_index(np.inf*np.dot(-Y_AXIS*(half_container_width_y+0.2),np.add(get_camera_position_vector(),Y_AXIS*(half_container_width_y+0.2))))),
                    # Wall([[-(half_container_width_x+0.2),(half_container_width_y+0.2),-half_container_height],[-(half_container_width_x+0.2),(half_container_width_y+0.2),half_container_height],[(half_container_width_x+0.2),(half_container_width_y+0.2),half_container_height],[(half_container_width_x+0.2),(half_container_width_y+0.2),-half_container_height]]).set_fill(color=DARK_BROWN,opacity=1).add_updater(lambda w: w.set_z_index(np.inf*np.dot(Y_AXIS*half_container_width_y,np.add(get_camera_position_vector(),-Y_AXIS*half_container_width_y)))),
                    # Rectangle(
                    #     width=0.2,
                    #     height=2*half_container_width_y+0.4,
                    #     stroke_width=0
                    # ).shift([half_container_width_x+0.1,0,half_container_height]).set_sheen(-0.25).set_fill(color=GREEN_E.darker(),opacity=1),
                    # Rectangle(
                    #     width=2*half_container_width_x+0.2,
                    #     height=0.2,
                    #     stroke_width=0
                    # ).shift([0,half_container_width_y+0.1,half_container_height]).set_sheen(-0.25).set_fill(color=GREEN_E.darker(),opacity=1),
                    # Rectangle(
                    #     width=0.2,
                    #     height=2*half_container_width_y+0.4,
                    #     stroke_width=0
                    # ).shift([-(half_container_width_x+0.1),0,half_container_height]).set_sheen(-0.25).set_fill(color=GREEN_E.darker(),opacity=1),
                    # Rectangle(
                    #     width=2*half_container_width_x+0.2,
                    #     height=0.2,
                    #     stroke_width=0
                    # ).shift([0,-(half_container_width_y+0.1),half_container_height]).set_sheen(-0.25).set_fill(color=GREEN_E.darker(),opacity=1),
                ]
        elif (container_shape=="cylinder"):
            cylinder=[
                Shell([
                    ArcBetweenPoints(start=[-container_radius,0,-half_container_height],end=[-container_radius,0,half_container_height],angle=0),
                    ArcBetweenPoints(start=[-container_radius,0,half_container_height],end=[container_radius,0,half_container_height],arc_center=[0,0,half_container_height],radius=container_radius),
                    ArcBetweenPoints(start=[container_radius,0,half_container_height],end=[container_radius,0,-half_container_height],angle=0),
                    ArcBetweenPoints(start=[container_radius,0,-half_container_height],end=[-container_radius,0,-half_container_height],arc_center=[0,0,-half_container_height],radius=-container_radius)
                ],radius=container_radius,semi_height=half_container_height,orientation=1),
                Shell([
                    ArcBetweenPoints(start=[-container_radius,0,-half_container_height],end=[-container_radius,0,half_container_height],angle=0),
                    ArcBetweenPoints(start=[-container_radius,0,half_container_height],end=[container_radius,0,half_container_height],arc_center=[0,0,half_container_height],radius=-container_radius),
                    ArcBetweenPoints(start=[container_radius,0,half_container_height],end=[container_radius,0,-half_container_height],angle=0),
                    ArcBetweenPoints(start=[container_radius,0,-half_container_height],end=[-container_radius,0,-half_container_height],arc_center=[0,0,-half_container_height],radius=container_radius)
                ],radius=container_radius,semi_height=half_container_height,orientation=-1),
                Lid(half_container_height, container_radius).add_updater(lambda w: w.set_z_index(np.inf*np.dot(Z_AXIS*half_container_height,np.add(get_camera_position_vector(),-Z_AXIS*half_container_height)))),
                Lid(-half_container_height, container_radius).add_updater(lambda w: w.set_z_index(np.inf*np.dot(-Z_AXIS*half_container_height,np.add(get_camera_position_vector(),Z_AXIS*half_container_height)))),
            ]
        elif (container_shape=="sphere"):
            globe=[
                Globe(container_radius).set_z_index(np.inf).add_updater(lambda g: g.set_sheen(container_sheen*(np.cos(PI/4)*np.cos(get_camera_phi())*np.sin(get_camera_theta())+np.sin(PI/4)*np.sin(get_camera_phi())),np.add(X_AXIS*np.cos(PI/4)*np.cos(get_camera_theta()),Y_AXIS*(np.cos(PI/4)*np.cos(get_camera_phi())*np.sin(get_camera_theta())+np.sin(PI/4)*np.sin(get_camera_phi()))))),
                Globe(container_radius).set_z_index(-np.inf).add_updater(lambda g: g.set_sheen(-0.5*container_sheen*(np.cos(PI/4)*np.cos(PI-get_camera_phi())*np.sin(-get_camera_theta())+np.sin(PI/4)*np.sin(PI-get_camera_phi())),np.add(X_AXIS*np.cos(PI/4)*np.cos(-get_camera_theta()),Y_AXIS*(np.cos(PI/4)*np.cos(PI-get_camera_phi())*np.sin(-get_camera_theta())+np.sin(PI/4)*np.sin(PI-get_camera_phi())))))
            ]
            globe[0].add_updater(lambda g: g.update_apparent_radius())
            globe[1].add_updater(lambda g: g.update_apparent_radius())

        #################
        ### Contents: ###
        #################
        
        particles=[]
        colors=color_gradient([PURE_RED,ORANGE,YELLOW,PURE_GREEN,XKCD.CLEARBLUE,XKCD.VIOLET,PURE_RED][:(num_particles+1)],num_particles+1)
        
        if newtons_cradle_on:
            particles = [
                Particle(scene=self,radius=particle_radius,mass=particle_mass,color=WHITE,energy=0,spawn_point=[0,-4.75,0]).accelerate([0,5,0]).set_size(mass=particle_mass*3),
                Particle(scene=self,radius=particle_radius,mass=particle_mass*0.25,color=XKCD.VIOLET,energy=0,spawn_point=[0,-1,0]).set_size(mass=particle_mass*0.25),
                # Particle(scene=self,radius=particle_radius,mass=particle_mass,color=XKCD.CLEARBLUE,energy=0,spawn_point=[0,-0.5,0]),
                Particle(scene=self,radius=particle_radius,mass=particle_mass,color=PURE_GREEN,energy=0,spawn_point=[0,0,0]),
                # Particle(scene=self,radius=particle_radius,mass=particle_mass,color=YELLOW,energy=0,spawn_point=[0,0.5,0]),
                Particle(scene=self,radius=particle_radius,mass=particle_mass,color=ORANGE,energy=0,spawn_point=[0,1,0]),
                Particle(scene=self,radius=particle_radius,mass=particle_mass*4,color=PURE_RED,energy=0,spawn_point=[0,1.7,0]).set_size(mass=particle_mass*4)
            ]
        else:
            energy_partitions=np.sort(np.append(rng.uniform(size=num_particles-1,low=0,high=total_kinetic_energy),[0,total_kinetic_energy]))
            for i in range(num_particles):
                current_energy=energy_partitions[i+1]-energy_partitions[i]
                # current_energy=total_kinetic_energy/num_particles
                new_particle=Particle(scene=self,radius=particle_radius,mass=particle_mass,color=particle_color,energy=current_energy)
                particles.append(new_particle)

        # particles.append(Particle(radius=particle_radius,mass=particle_mass).set_z_index(np.inf).set_color(GREEN))

        if large_particle_on:
            large_particle=Particle(radius=large_particle_radius,mass=large_particle_mass,color=large_particle_color)
            large_particle.move_to(ORIGIN).set_velocity(ORIGIN)
            particles.append(large_particle)
            def brownian_motion():
                for particle in particles:
                    detect_particle_collision(large_particle, particle)
            large_particle.add_updater(lambda x: brownian_motion())

        aabb_tree = AABBTree()
        for particle in particles:
            particle.aabb=particle.get_aabb()
            aabb_tree.insert(particle.aabb)
            
        def draw_vectors():
            vectors=VGroup()
            for particle in particles:
                # vectors.add(Arrow(start=particle.get_center(),end=np.add(particle.get_center(),particle.velocity),stroke_width=5,tip_length=1).set_color(YELLOW))
                vectors.add(Arrow(start=particle.get_center(),end=np.add(particle.get_center(),np.array([particle.velocity[0]/temperature,0,0])),stroke_width=5,tip_length=0.1).set_color(PURE_RED))
                vectors.add(Arrow(start=particle.get_center(),end=np.add(particle.get_center(),np.array([0,particle.velocity[1]/temperature,0])),stroke_width=5,tip_length=0.1).set_color(PURE_GREEN))
                vectors.add(Arrow(start=particle.get_center(),end=np.add(particle.get_center(),np.array([0,0,particle.velocity[2]/temperature])),stroke_width=5,tip_length=0.1).set_color(PURE_BLUE))
            return vectors

        #######################
        ### Initialization: ###
        #######################

        bucket_size=10*np.sqrt(temperature)
        graph_x_range=250*np.sqrt(temperature)
        graph_y_range=0.015/np.sqrt(temperature)
        energy_x_range=7*(k_B*temperature)
        energy_y_range=0.6/(k_B*temperature)

        if display_chart:
            axes_MB=Axes(x_range=(0,graph_x_range,graph_x_range/5),y_range=(0,graph_y_range,graph_y_range/3),x_length=4,y_length=2,tips=False,axis_config={"include_numbers": False}).shift(RIGHT*4.75).shift(UP*1.5)
            axes_energy=Axes(x_range=(0,energy_x_range,energy_x_range/5),y_range=(0,energy_y_range,energy_y_range/3),x_length=4,y_length=2,tips=False,axis_config={"include_numbers": False}).shift(RIGHT*4.75).shift(DOWN*1.5)
            axes_X=Axes(x_range=(-2*graph_x_range/3,2*graph_x_range/3,graph_x_range/3),y_range=(0,0.6*graph_y_range,0.2*graph_y_range),x_length=3,y_length=1,tips=False,axis_config={"include_numbers": False}).shift(LEFT*4.75).shift(UP*2)
            axes_Y=Axes(x_range=(-2*graph_x_range/3,2*graph_x_range/3,graph_x_range/3),y_range=(0,0.6*graph_y_range,0.2*graph_y_range),x_length=3,y_length=1,tips=False,axis_config={"include_numbers": False}).shift(LEFT*4.75)
            axes_Z=Axes(x_range=(-2*graph_x_range/3,2*graph_x_range/3,graph_x_range/3),y_range=(0,0.6*graph_y_range,0.2*graph_y_range),x_length=3,y_length=1,tips=False,axis_config={"include_numbers": False}).shift(LEFT*4.75).shift(DOWN*2)
            
            maxwell_boltzmann_graph=axes_MB.plot(lambda x: maxwell_boltzmann(x, particle_mass, temperature), color=PURE_RED)
            MB_energy_graph=axes_energy.plot(lambda x: MB_energy(x, temperature), color=PURE_RED)
            gaussian_X=axes_X.plot(lambda x: gaussian(x, particle_mass, temperature), color=PURE_RED)
            gaussian_Y=axes_Y.plot(lambda x: gaussian(x, particle_mass, temperature), color=PURE_RED)
            gaussian_Z=axes_Z.plot(lambda x: gaussian(x, particle_mass, temperature), color=PURE_RED)
            
            particle_speed_distribution=always_redraw(lambda: axes_MB.plot(lambda x: get_frequency_plotter(particles,"get_speed",bucket_size)(x)/(num_particles*bucket_size),use_smoothing=False,color=PURE_GREEN))
            particle_speeds_histogram=always_redraw(lambda: axes_MB.get_riemann_rectangles(particle_speed_distribution, x_range=(0,graph_x_range), dx=bucket_size, input_sample_type='center', stroke_width=0, fill_opacity=0.7, color=(PURE_BLUE,BLUE,BLUE), show_signed_area=False, bounded_graph=None, width_scale_factor=1.0155))
            
            particle_energy_distribution=always_redraw(lambda: axes_energy.plot(lambda x: get_frequency_plotter(particles,"get_kinetic_energy",energy_x_range/25)(x)/(num_particles*energy_x_range/25),use_smoothing=False,color=PURE_GREEN))
            particle_energies_histogram=always_redraw(lambda: axes_energy.get_riemann_rectangles(particle_energy_distribution, x_range=(0,energy_x_range), dx=energy_x_range/25, input_sample_type='center', stroke_width=0, fill_opacity=0.7, color=(ManimColor('#9A0EEA'),ManimColor('#CAA0FF'),ManimColor('#CAA0FF')), show_signed_area=False, bounded_graph=None, width_scale_factor=1.0155))
            
            particle_v_X_distribution=always_redraw(lambda: axes_X.plot(lambda x: get_frequency_plotter(particles,"get_x_velocity",graph_x_range/15)(x)/(num_particles*graph_x_range/15),use_smoothing=False,color=PURE_GREEN))
            particle_v_X_histogram=always_redraw(lambda: axes_X.get_riemann_rectangles(particle_v_X_distribution, x_range=(-2*graph_x_range/3,2*graph_x_range/3), dx=graph_x_range/15, input_sample_type='center', stroke_width=0, fill_opacity=0.7, color=(YELLOW_D,YELLOW_B,YELLOW_A), show_signed_area=False, bounded_graph=None, width_scale_factor=1.0155))
            particle_v_Y_distribution=always_redraw(lambda: axes_Y.plot(lambda x: get_frequency_plotter(particles,"get_y_velocity",graph_x_range/15)(x)/(num_particles*graph_x_range/15),use_smoothing=False,color=PURE_GREEN))
            particle_v_Y_histogram=always_redraw(lambda: axes_Y.get_riemann_rectangles(particle_v_Y_distribution, x_range=(-2*graph_x_range/3,2*graph_x_range/3), dx=graph_x_range/15, input_sample_type='center', stroke_width=0, fill_opacity=0.7, color=(PURE_RED,RED_B,RED_A), show_signed_area=False, bounded_graph=None, width_scale_factor=1.0155))
            particle_v_Z_distribution=always_redraw(lambda: axes_Z.plot(lambda x: get_frequency_plotter(particles,"get_z_velocity",graph_x_range/15)(x)/(num_particles*graph_x_range/15),use_smoothing=False,color=PURE_GREEN))
            particle_v_Z_histogram=always_redraw(lambda: axes_Z.get_riemann_rectangles(particle_v_Z_distribution, x_range=(-2*graph_x_range/3,2*graph_x_range/3), dx=graph_x_range/15, input_sample_type='center', stroke_width=0, fill_opacity=0.7, color=(PURE_GREEN,GREEN_B,GREEN_A), show_signed_area=False, bounded_graph=None, width_scale_factor=1.0155))
            
        # self.add_fixed_in_frame_mobjects(Text(str(round((10**27)*total_kinetic_energy,2))).shift(LEFT*4).shift(UP))
        # def sum_kinetic_energy(total, particle):
        #     return total + particle.get_kinetic_energy()
        # self.add_fixed_in_frame_mobjects(always_redraw(lambda: Text(str(round((10**27)*reduce(sum_kinetic_energy, particles, 0), 2))).shift(RIGHT*4).shift(UP)))
        # def sum_potential_energy(total, particle):
        #     return total + particle.get_potential_energy()
        # self.add_fixed_in_frame_mobjects(always_redraw(lambda: Text(str(round((10**27)*reduce(sum_potential_energy, particles, 0), 2))).shift(RIGHT*4).shift(DOWN)))
        # def sum_energy(total, particle):
        #     return total + particle.get_energy()
        # self.add_fixed_in_frame_mobjects(always_redraw(lambda: Text(str(round((10**27)*reduce(sum_energy, particles, 0), 2))).shift(LEFT*4).shift(DOWN)))
        
        # positions = VGroup(*(always_redraw(lambda: Text(str(round(particles[i].get_y(),3)),color=particles[i].color).shift(LEFT*4).shift(DOWN*(i-3))) for i in range(len(particles))))
        # self.add_fixed_in_frame_mobjects(positions)
        
        # position1 = ((always_redraw(lambda: Text(str(particles[0].get_y()),color=particles[0].color).shift(LEFT*4).shift(DOWN*(0-3)))))
        # self.add_fixed_in_frame_mobjects(position1)
        # position2 = ((always_redraw(lambda: Text(str(particles[1].get_y()),color=particles[1].color).shift(LEFT*4).shift(DOWN*(1-3)))))
        # self.add_fixed_in_frame_mobjects(position2)
        # position3 = ((always_redraw(lambda: Text(str(particles[2].get_y()),color=particles[2].color).shift(LEFT*4).shift(DOWN*(2-3)))))
        # self.add_fixed_in_frame_mobjects(position3)
        # position4 = ((always_redraw(lambda: Text(str(particles[3].get_y()),color=particles[3].color).shift(LEFT*4).shift(DOWN*(3-3)))))
        # self.add_fixed_in_frame_mobjects(position4)
        # position5 = ((always_redraw(lambda: Text(str(particles[4].get_y()),color=particles[4].color).shift(LEFT*4).shift(DOWN*(4-3)))))
        # self.add_fixed_in_frame_mobjects(position5)
        # position6 = ((always_redraw(lambda: Text(str(particles[5].get_y()),color=particles[5].color).shift(LEFT*4).shift(DOWN*(5-3)))))
        # self.add_fixed_in_frame_mobjects(position6)
        # position7 = ((always_redraw(lambda: Text(str(particles[6].get_y()),color=particles[6].color).shift(LEFT*4).shift(DOWN*(6-3)))))
        # self.add_fixed_in_frame_mobjects(position7)

        if (container_shape=="box"): self.add(*walls)
        elif (container_shape=="cylinder"): self.add(*cylinder)
        elif (container_shape=="sphere"): self.add_fixed_orientation_mobjects(*globe)
        # self.add(always_redraw(lambda: Arrow(start=ORIGIN,end=get_camera_unit_vector(),color=PURE_GREEN).rotate(0.5*PI,Z_AXIS)))
        # self.add(Cube(side_length=2*half_container_height/num_divs))
        for particle in particles:
            self.add(particle)
            # self.add(particle.shadow)
        if collisions_on:
            particles[0].add_updater(lambda x, dt: check_AABB_collisions(aabb_tree, particles, dt))
            if wall_clacks_on or particle_clacks_on:
                self.add(clacks)
        piston=Piston(mass=200*Da,force=500*Da)
        if piston_on:
            self.add(piston)
        if trails_on:
            self.add(trails)
        if vectors_on: self.add(always_redraw(draw_vectors))
        self.renderer.camera.set_focal_distance(focal_distance)
        self.set_camera_orientation(theta=camera_initial_theta, phi=camera_initial_phi)
        self.begin_ambient_camera_rotation(rate=-camera_rotation_rate, about=camera_rotation_axis)
        if display_chart: self.add_fixed_in_frame_mobjects(
            axes_MB,
            axes_energy,
            axes_X, axes_Y, axes_Z,
            maxwell_boltzmann_graph, particle_speeds_histogram,
            MB_energy_graph,
            particle_energies_histogram,
            gaussian_X, particle_v_X_histogram,
            gaussian_Y, particle_v_Y_histogram,
            gaussian_Z, particle_v_Z_histogram
        )
        self.wait(duration)
        # self.interactive_embed()