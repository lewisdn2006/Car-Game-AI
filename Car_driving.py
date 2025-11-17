import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
RED = (255, 0, 0)

class Car:
    def __init__(self, x, y):
        # Load and scale the car image
        try:
            self.original_image = pygame.image.load("car.png").convert_alpha()
            # Scale the car to a reasonable size
            self.original_image = pygame.transform.scale(self.original_image, (40, 20))
            # Flip the car image horizontally to correct orientation
            self.original_image = pygame.transform.flip(self.original_image, True, False)
        except pygame.error:
            # If car.png is not found, create a simple rectangle
            self.original_image = pygame.Surface((40, 20))
            self.original_image.fill(RED)
            print("car.png not found, using red rectangle")
        
        self.image = self.original_image
        self.rect = self.image.get_rect()
        
        # Position and movement
        self.x = float(x)
        self.y = float(y)
        self.angle = -90.0  # Car's rotation angle (90 degrees anticlockwise)
        
        # Simple movement properties
        self.speed = 0.0  # Current speed
        self.max_speed = 8.0  # Maximum speed
        self.acceleration = 0.15  # How fast the car accelerates
        self.deceleration = 0.1  # How fast the car slows down
        self.turn_rate = 3.0  # How fast the car turns (for manual control)
        
        # Additional properties needed for AI training
        self.turn_speed = 0.0  # Current turning speed (used by AI)
        self.max_turn_speed = 5.0  # Maximum turn speed for AI
        self.turn_friction = 0.9  # Turn friction for AI
        
        # Collision - use rectangular hitbox matching image size
        self.width = 40
        self.height = 20
        
        # Starting position for reset
        self.start_x = x
        self.start_y = y
        
        # Ray casting for wall detection
        self.num_rays = 8
        self.ray_length = 2000  # Very large ray length to ensure they always hit walls
        self.ray_distances = [self.ray_length] * self.num_rays  # Store distances to walls
        
        # Collision toggle
        self.collision_enabled = True
        
    def update(self, keys_pressed, track):
        # Handle input and update speed
        if keys_pressed[pygame.K_UP]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif keys_pressed[pygame.K_DOWN]:
            self.speed = max(self.speed - self.acceleration * 2, -self.max_speed / 2)  # Reverse at half speed
        else:
            # Natural deceleration when no input
            if self.speed > 0:
                self.speed = max(0, self.speed - self.deceleration)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.deceleration)
        
        # Handle turning (only when moving)
        if abs(self.speed) > 0.1:  # Only turn when moving
            if keys_pressed[pygame.K_LEFT]:
                self.angle -= self.turn_rate
            elif keys_pressed[pygame.K_RIGHT]:
                self.angle += self.turn_rate
        
        # Normalize angle
        while self.angle > 180:
            self.angle -= 360
        while self.angle < -180:
            self.angle += 360
        
        # Update position based on speed and angle
        car_angle_rad = math.radians(self.angle)
        self.x += math.cos(car_angle_rad) * self.speed
        self.y += math.sin(car_angle_rad) * self.speed
            
        # Check for wall collision using rectangular hitbox (if collision is enabled)
        if not self.collision_enabled or not track.check_rectangle_collision(self.x, self.y, self.width, self.height, self.angle):
            # Check for checkpoint collision at new position
            if track.check_checkpoint_collision(self.x, self.y, self.width, self.height, self.angle):
                track.hit_checkpoint()
        elif self.collision_enabled:
            # Collision detected and enabled, reset car to starting position and reset checkpoints
            self.reset_position()
            track.reset_checkpoints()
        
        # Update ray casting distances
        self.update_ray_distances(track)
        
        # Rotate the car image
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
    def reset_position(self):
        """Reset car to starting position and stop all movement"""
        self.x = self.start_x
        self.y = self.start_y
        self.angle = -90.0
        self.speed = 0.0
        # Reset turn speed for AI compatibility
        self.turn_speed = 0.0
    
    def toggle_collision(self):
        """Toggle collision detection on/off"""
        self.collision_enabled = not self.collision_enabled
        print(f"Collision detection: {'ON' if self.collision_enabled else 'OFF'}")
    
    def update_ray_distances(self, track):
        """Update the distances from car to walls using ray casting"""
        # Reset all distances to maximum
        self.ray_distances = [self.ray_length] * self.num_rays
        
        # Cast rays in 8 directions (45 degrees apart)
        for i in range(self.num_rays):
            # Calculate ray angle (relative to world, not car orientation)
            ray_angle = self.angle + (i * 45)  # 360/8 = 45 degrees apart
            ray_angle_rad = math.radians(ray_angle)
            
            # Cast ray and find intersection with walls
            min_distance = self.ray_length
            
            # Check intersection with all walls
            for wall in track.walls:
                distance = self.ray_intersect_wall(ray_angle_rad, wall)
                if distance is not None and distance < min_distance:
                    min_distance = distance
            
            self.ray_distances[i] = min_distance
    
    def ray_intersect_wall(self, ray_angle_rad, wall):
        """Calculate intersection distance between a ray and a wall"""
        # Ray starting point (car center)
        ray_start_x = self.x
        ray_start_y = self.y
        
        # Ray direction
        ray_dx = math.cos(ray_angle_rad)
        ray_dy = math.sin(ray_angle_rad)
        
        # Ray end point
        ray_end_x = ray_start_x + ray_dx * self.ray_length
        ray_end_y = ray_start_y + ray_dy * self.ray_length
        
        # Wall points
        wall_x1, wall_y1 = wall[0]
        wall_x2, wall_y2 = wall[1]
        
        # Line intersection calculation
        # Ray: (ray_start_x, ray_start_y) to (ray_end_x, ray_end_y)
        # Wall: (wall_x1, wall_y1) to (wall_x2, wall_y2)
        
        ray_dx_total = ray_end_x - ray_start_x
        ray_dy_total = ray_end_y - ray_start_y
        wall_dx = wall_x2 - wall_x1
        wall_dy = wall_y2 - wall_y1
        
        denominator = ray_dx_total * wall_dy - ray_dy_total * wall_dx
        
        # Lines are parallel
        if abs(denominator) < 1e-10:
            return None
        
        # Calculate intersection parameters
        t = ((wall_x1 - ray_start_x) * wall_dy - (wall_y1 - ray_start_y) * wall_dx) / denominator
        u = ((wall_x1 - ray_start_x) * ray_dy_total - (wall_y1 - ray_start_y) * ray_dx_total) / denominator
        
        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate distance from car to intersection point
            intersect_x = ray_start_x + t * ray_dx_total
            intersect_y = ray_start_y + t * ray_dy_total
            distance = math.sqrt((intersect_x - ray_start_x)**2 + (intersect_y - ray_start_y)**2)
            return distance
        
        return None
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)
        
        # Draw rays for wall detection
        self.draw_rays(screen)
        
        # Draw speed indicator
        speed_text = f"Speed: {abs(self.speed):.1f}"
        font = pygame.font.Font(None, 24)
        text_surface = font.render(speed_text, True, WHITE)
        screen.blit(text_surface, (10, 10))
        
        # Draw direction indicator
        direction = "Forward" if self.speed > 0 else "Reverse" if self.speed < 0 else "Stopped"
        direction_surface = font.render(f"Direction: {direction}", True, WHITE)
        screen.blit(direction_surface, (10, 35))
    
    def draw_rays(self, screen):
        """Draw rays emanating from the car"""
        for i in range(self.num_rays):
            # Calculate ray angle (relative to world, not car orientation)
            ray_angle = self.angle + (i * 45)  # 45 degrees apart
            ray_angle_rad = math.radians(ray_angle)
            
            # Calculate ray end point based on detected distance
            distance = self.ray_distances[i]
            end_x = self.x + math.cos(ray_angle_rad) * distance
            end_y = self.y + math.sin(ray_angle_rad) * distance
            
            # Draw ray line (green since they should always hit a wall now)
            color = (0, 255, 0)  # Always green since rays should always hit walls
            pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 2)
            
            # Draw small circle at ray end
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)

class Track:
    def __init__(self):
        # Define track as a series of wall segments (line segments)
        # Each wall is defined as ((x1, y1), (x2, y2))
        self.walls = []
        
        # Create a simple rectangular track for testing
        track_margin = 100
        track_width = SCREEN_WIDTH - 2 * track_margin
        track_height = SCREEN_HEIGHT - 2 * track_margin
        
        # New walls from user
        walls = [
            # From sequence 1
            ((230, 422), (235, 323)),
            ((235, 323), (273, 234)),
            ((273, 234), (353, 190)),
            ((353, 190), (469, 168)),
            ((469, 168), (574, 162)),
            ((574, 162), (658, 163)),
            ((658, 163), (788, 169)),
            ((788, 169), (857, 193)),
            ((857, 193), (888, 250)),
            ((888, 250), (892, 335)),
            ((892, 335), (865, 388)),
            ((865, 388), (826, 436)),
            ((826, 436), (816, 486)),
            ((816, 486), (848, 539)),
            ((848, 539), (907, 586)),
            ((907, 586), (919, 653)),
            ((919, 653), (880, 700)),
            ((880, 700), (803, 736)),
            ((803, 736), (716, 741)),
            ((716, 741), (648, 739)),
            ((648, 739), (575, 741)),
            ((575, 741), (519, 739)),
            ((519, 739), (454, 722)),
            ((454, 722), (361, 687)),
            ((361, 687), (331, 638)),
            ((331, 638), (281, 587)),
            ((281, 587), (258, 522)),
            ((258, 522), (242, 475)),
            ((242, 475), (230, 422)),
            # From sequence 2
            ((121, 425), (120, 321)),
            ((120, 321), (129, 244)),
            ((129, 244), (160, 175)),
            ((160, 175), (216, 118)),
            ((216, 118), (276, 86)),
            ((276, 86), (399, 60)),
            ((399, 60), (513, 47)),
            ((513, 47), (637, 42)),
            ((637, 42), (779, 53)),
            ((779, 53), (869, 60)),
            ((869, 60), (952, 100)),
            ((952, 100), (998, 161)),
            ((998, 161), (1025, 255)),
            ((1025, 255), (1026, 340)),
            ((1026, 340), (1018, 402)),
            ((1018, 402), (987, 440)),
            ((987, 440), (941, 466)),
            ((941, 466), (955, 492)),
            ((955, 492), (984, 536)),
            ((984, 536), (1034, 567)),
            ((1034, 567), (1050, 634)),
            ((1050, 634), (1054, 733)),
            ((1054, 733), (1010, 781)),
            ((1010, 781), (925, 784)),
            ((925, 784), (304, 781)),
            ((304, 781), (241, 743)),
            ((241, 743), (211, 688)),
            ((211, 688), (180, 616)),
            ((180, 616), (155, 572)),
            ((155, 572), (135, 518)),
            ((135, 518), (121, 425)),
        ]
        
        # Add all walls to the track
        self.walls.extend(walls)
        
        # Checkpoints along the custom track - defined as lines crossing the track
        # Each checkpoint is ((x1, y1), (x2, y2)) representing a line
        self.checkpoints = [
            ((106, 327), (241, 349)),
            ((149, 185), (272, 257)),
            ((274, 72), (328, 207)),
            ((405, 52), (425, 178)),
            ((530, 38), (530, 169)),
            ((645, 33), (644, 165)),
            ((735, 38), (728, 179)),
            ((818, 48), (790, 174)),
            ((959, 97), (845, 205)),
            ((1031, 218), (885, 244)),
            ((1038, 304), (876, 307)),
            ((1028, 398), (862, 359)),
            ((968, 467), (825, 403)),
            ((954, 484), (795, 501)),
            ((985, 521), (865, 584)),
            ((1034, 554), (898, 645)),
            ((1060, 676), (893, 668)),
            ((1049, 756), (870, 688)),
            ((831, 711), (914, 780)),
            ((747, 735), (753, 780)),
            ((627, 732), (629, 781)),
            ((542, 732), (542, 782)),
            ((450, 712), (423, 782)),
            ((363, 679), (274, 770)),
            ((318, 612), (183, 667)),
            ((279, 553), (132, 563)),
            ((245, 462), (106, 469))
        ]
        
        # Checkpoint system
        self.current_checkpoint = 0  # Index of the active checkpoint (0-3)
        self.checkpoints_hit = [False] * len(self.checkpoints)  # Track which checkpoints have been hit
        
    def draw(self, screen):
        # Draw track surface (grass background)
        screen.fill((34, 139, 34))  # Forest green
        
        # Draw all walls as white lines
        for wall in self.walls:
            pygame.draw.line(screen, WHITE, wall[0], wall[1], 4)
        
        # Draw checkpoints as lines crossing the track
        for i, checkpoint in enumerate(self.checkpoints):
            # Active checkpoint is red, others are yellow
            if i == self.current_checkpoint:
                color = RED  # Active checkpoint
            else:
                color = (255, 255, 0)  # Yellow for inactive checkpoints
            
            # Draw the checkpoint line with thick width
            pygame.draw.line(screen, color, checkpoint[0], checkpoint[1], 6)
            
            # Draw checkpoint number near the middle of the line
            line_center_x = (checkpoint[0][0] + checkpoint[1][0]) // 2
            line_center_y = (checkpoint[0][1] + checkpoint[1][1]) // 2
            
            font = pygame.font.Font(None, 24)
            text = font.render(str(i + 1), True, WHITE)
            
            # Offset text so it doesn't overlap the line
            text_offset = 20
            text_x = line_center_x + text_offset
            text_y = line_center_y - text_offset
            
            # Draw black background for text readability
            text_rect = text.get_rect(center=(text_x, text_y))
            pygame.draw.rect(screen, BLACK, text_rect.inflate(6, 2))
            screen.blit(text, text_rect)
    
    def check_rectangle_collision(self, center_x, center_y, width, height, angle):
        """Check if a rotated rectangle collides with any wall"""
        # Calculate the four corners of the rectangle
        half_width = width / 2
        half_height = height / 2
        
        # Corner offsets relative to center (before rotation)
        corners = [
            (-half_width, -half_height),  # Top-left
            (half_width, -half_height),   # Top-right
            (half_width, half_height),    # Bottom-right
            (-half_width, half_height)    # Bottom-left
        ]
        
        # Rotate corners based on angle
        rad_angle = math.radians(angle)
        cos_angle = math.cos(rad_angle)
        sin_angle = math.sin(rad_angle)
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            # Rotate point around origin
            rotated_x = corner_x * cos_angle - corner_y * sin_angle
            rotated_y = corner_x * sin_angle + corner_y * cos_angle
            
            # Translate to rectangle's position
            world_x = center_x + rotated_x
            world_y = center_y + rotated_y
            
            rotated_corners.append((world_x, world_y))
        
        # Check if any corner is too close to any wall
        for corner in rotated_corners:
            for wall in self.walls:
                if self.point_to_line_distance(corner, wall[0], wall[1]) < 2:  # Small tolerance
                    return True
        
        # Check if any rectangle edge intersects with any wall
        for i in range(4):
            rect_edge_start = rotated_corners[i]
            rect_edge_end = rotated_corners[(i + 1) % 4]
            
            for wall in self.walls:
                if self.line_intersects_line(rect_edge_start, rect_edge_end, wall[0], wall[1]):
                    return True
        
        return False
    
    def line_intersects_line(self, line1_start, line1_end, line2_start, line2_end):
        """Check if two line segments intersect"""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end
        
        # Calculate the direction of the lines
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Lines are parallel
        if abs(denom) < 1e-10:
            return False
        
        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection point lies on both line segments
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def check_checkpoint_collision(self, car_x, car_y, car_width, car_height, car_angle):
        """Check if the car has hit the current active checkpoint"""
        if self.current_checkpoint >= len(self.checkpoints):
            return False
        
        # Get the current active checkpoint line
        checkpoint_line = self.checkpoints[self.current_checkpoint]
        
        # Get car's corners for precise collision detection
        half_width = car_width / 2
        half_height = car_height / 2
        
        # Corner offsets relative to center (before rotation)
        corners = [
            (-half_width, -half_height),  # Top-left
            (half_width, -half_height),   # Top-right
            (half_width, half_height),    # Bottom-right
            (-half_width, half_height)    # Bottom-left
        ]
        
        # Rotate corners based on car's angle
        rad_angle = math.radians(car_angle)
        cos_angle = math.cos(rad_angle)
        sin_angle = math.sin(rad_angle)
        
        car_corners = []
        for corner_x, corner_y in corners:
            # Rotate point around origin
            rotated_x = corner_x * cos_angle - corner_y * sin_angle
            rotated_y = corner_x * sin_angle + corner_y * cos_angle
            
            # Translate to car's position
            world_x = car_x + rotated_x
            world_y = car_y + rotated_y
            
            car_corners.append((world_x, world_y))
        
        # Check if any car edge intersects with the checkpoint line
        for i in range(4):
            car_edge_start = car_corners[i]
            car_edge_end = car_corners[(i + 1) % 4]
            
            if self.line_intersects_line(car_edge_start, car_edge_end, checkpoint_line[0], checkpoint_line[1]):
                return True
        
        # Also check if car center is very close to the checkpoint line
        if self.point_to_line_distance((car_x, car_y), checkpoint_line[0], checkpoint_line[1]) < 20:
            return True
        
        return False
    
    def hit_checkpoint(self):
        """Called when the car hits the current active checkpoint"""
        # Move to next checkpoint
        self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)
        #print(f"Checkpoint hit! Next checkpoint: {self.current_checkpoint + 1}")
    
    def reset_checkpoints(self):
        """Reset checkpoint system to start from checkpoint 1"""
        self.current_checkpoint = 0
        self.checkpoints_hit = [False] * len(self.checkpoints)
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the shortest distance from a point to a line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line is just a point
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t represents position along the line
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        
        # Clamp t to the line segment
        t = max(0, min(1, t))
        
        # Find the closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Return distance to closest point
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

class Game:
    def __init__(self):
        print("Initializing pygame display...")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Car Racing Game")
        self.clock = pygame.time.Clock()
        
        print("Creating game objects...")
        # Game objects
        self.car = Car(175, 425)  # Start position for new track layout
        self.track = Track()
        
        # Game state
        self.running = True
        
        # Camera for following the car (optional)
        self.camera_x = 0
        self.camera_y = 0
        self.camera_follow = False  # Set to True for following camera
        print("Game initialization complete")
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset car position
                    self.car.reset_position()
                    # Reset checkpoint system
                    self.track.reset_checkpoints()
                elif event.key == pygame.K_t:
                    # Toggle collision detection
                    self.car.toggle_collision()
                elif event.key == pygame.K_c:
                    # Toggle camera follow
                    self.camera_follow = not self.camera_follow
    
    def update(self):
        keys_pressed = pygame.key.get_pressed()
        self.car.update(keys_pressed, self.track)  # Pass track for collision detection
        
        # Update camera if following
        if self.camera_follow:
            target_x = self.car.x - SCREEN_WIDTH // 2
            target_y = self.car.y - SCREEN_HEIGHT // 2
            self.camera_x += (target_x - self.camera_x) * 0.1
            self.camera_y += (target_y - self.camera_y) * 0.1
    
    def draw(self):
        # Clear screen - the track will draw its own background
        # self.screen.fill(DARK_GRAY)  # Commented out since track draws background
        
        # Apply camera offset if following
        if self.camera_follow:
            # This would require adjusting all draw positions by camera offset
            # For simplicity, we'll keep camera disabled for now
            pass
        
        # Draw track
        self.track.draw(self.screen)
        
        # Draw car
        self.car.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def draw_ui(self):
        # Instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "Arrow Keys: Drive the car",
            "Up: Accelerate Forward",
            "Down: Reverse/Brake",
            "Left/Right: Turn (only when moving)",
            "R: Reset position",
            "T: Toggle collision detection",
            "ESC: Quit",
            f"Angle: {self.car.angle:.1f}°",
            f"Collision: {'ON' if self.car.collision_enabled else 'OFF'}"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = font.render(instruction, True, WHITE)
            self.screen.blit(text_surface, (10, 40 + i * 25))
        
        # Speed bar
        speed_ratio = abs(self.car.speed) / self.car.max_speed
        bar_width = 200
        bar_height = 20
        bar_x = SCREEN_WIDTH - bar_width - 20
        bar_y = 20
        
        # Background
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Speed fill
        fill_width = int(bar_width * speed_ratio)
        color = RED if speed_ratio > 0.8 else (255, 255, 0) if speed_ratio > 0.5 else GREEN
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
        
        # Speed label
        speed_text = font.render("Speed", True, WHITE)
        self.screen.blit(speed_text, (bar_x, bar_y - 25))
        
        # Draw ray distances in top right corner
        self.draw_ray_distances()
    
    def draw_ray_distances(self):
        """Draw the ray distances in the top right corner"""
        font = pygame.font.Font(None, 20)
        start_x = SCREEN_WIDTH - 200
        start_y = 60
        
        # Title
        title_text = font.render("Ray Distances:", True, WHITE)
        self.screen.blit(title_text, (start_x, start_y))
        
        # Draw distances for each ray
        for i in range(self.car.num_rays):
            distance = self.car.ray_distances[i]
            # Calculate ray angle for display
            ray_angle = (self.car.angle + (i * 45)) % 360
            
            # Format distance and angle
            distance_text = f"Ray {i+1} ({ray_angle:.0f}°): {distance:.1f}"
            
            # Color code based on distance (red = close, green = far)
            if distance < 30:
                color = RED
            elif distance < 60:
                color = (255, 255, 0)  # Yellow
            else:
                color = GREEN
            
            text_surface = font.render(distance_text, True, color)
            self.screen.blit(text_surface, (start_x, start_y + 25 + i * 18))
    
    def run(self):
        print("Starting game loop...")
        while self.running:
            try:
                self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(FPS)
            except Exception as e:
                print(f"Error in game loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("Game loop ended, cleaning up...")
        pygame.quit()
        sys.exit()

def main():
    print("Starting car racing game...")
    try:
        game = Game()
        print("Game initialized successfully")
        game.run()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to close...")

if __name__ == "__main__":
    main()