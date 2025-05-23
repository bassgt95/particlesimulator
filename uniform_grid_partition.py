import numpy as np

def check_particle_collisions(num_divs):
            collisions=[]
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
            # for row in grid:
            #     for column in row:
            #         for cell in column: