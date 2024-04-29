# Imports
import arcade
import random

# Constants
SPRITE_SIZE = 16
MAZE_SIZE = 20
SCALING = 2.0 # I predict unpredictable behavior if this does not evenly divide both sprite and maze size
SCALING_INT = int(SCALING) 
SCREEN_WIDTH = SPRITE_SIZE * MAZE_SIZE * SCALING_INT
SCREEN_HEIGHT = SPRITE_SIZE * MAZE_SIZE * SCALING_INT
SCREEN_TITLE = "Torchlight"

class Torchlight(arcade.Window):

    def __init__(self, width, height, title):
        """
        Initialize the game
        """
        super().__init__(width, height, title)

        # Set up the empty sprite lists
        self.wall_list = arcade.SpriteList(use_spatial_hash=True)
        self.fog_list = arcade.SpriteList()
        self.all_sprites = arcade.SpriteList()

        self.physics_engine = None
        self.paused = False
    
    def setup(self):
        """
        Get the game ready to play
        """

        # Set the background color
        arcade.set_background_color((219, 168, 140))

        # Set up the player
        self.player = arcade.Sprite("./images/skull_v2_1.png", SCALING) #TODO Set hitbox here to be 16x16; will center the sprite and fix shunting?
        self.player.top = self.height
        self.player.left = 0
        self.all_sprites.append(self.player)

        self.end_tile = arcade.Sprite("./images/end_tile.png", SCALING)
        self.end_tile.left = 0
        self.end_tile.bottom = 0
        self.all_sprites.append(self.end_tile)        

        # Set up the maze
        def create_maze():
            """
            Credit for maze generation technique goes to github.com/ChickenSlayer3000/Random-Maze-Generator
            Had to modify it to work with my own stuff, and I also changed some variables for readability; 
            others I chose not to mess with.
            For deciphering, I understand scr and scc to be StartingCellRow and StartingCellColumn;
            similar with Current and End.
            I hope to fully make my own at some point instead of using a modified version of this.           
            """

            ms = MAZE_SIZE # rows and columns
            visited_cells = []
            walls = []
            revisited_cells = []

            map = [['Blocked' for _ in range(ms)]for _ in range(ms)]

            def create():
                for row in range(ms):
                    for col in range(ms):
                        if map[row][col] == 'Open':
                            pass
                            # Do nothing for now; background has us covered here; may want to change this later when we add more hazards like pits
                        elif map[row][col] == 'Blocked':
                            sprite = arcade.Sprite("./images/wall_tile.png", SCALING)

                            x = col * SPRITE_SIZE * SCALING_INT
                            y = row * SPRITE_SIZE * SCALING_INT

                            sprite.left = x
                            sprite.bottom = y
                            self.wall_list.append(sprite)
                            self.all_sprites.append(sprite)

            def check_neighbours(ccr, ccc):
                neighbours = [[ccr, ccc-1, ccr-1, ccc-2, ccr, ccc-2, ccr+1, ccc-2, ccr-1, ccc-1, ccr+1, ccc-1], #left
                            [ccr, ccc+1, ccr-1, ccc+2, ccr, ccc+2, ccr+1, ccc+2, ccr-1, ccc+1, ccr+1, ccc+1], #right
                            [ccr-1, ccc, ccr-2, ccc-1, ccr-2, ccc, ccr-2, ccc+1, ccr-1, ccc-1, ccr-1, ccc+1], #top
                            [ccr+1, ccc, ccr+2, ccc-1, ccr+2, ccc, ccr+2, ccc+1, ccr+1, ccc-1, ccr+1, ccc+1]] #bottom
                visitable_neighbours = []           
                for i in neighbours:                                                                        #find neighbours to visit
                    if i[0] > 0 and i[0] < (ms-1) and i[1] > 0 and i[1] < (ms-1):
                        if map[i[2]][i[3]] == 'Open' or map[i[4]][i[5]] == 'Open' or map[i[6]][i[7]] == 'Open' or map[i[8]][i[9]] == 'Open' or map[i[10]][i[11]] == 'Open':
                            walls.append(i[0:2])                                                                                               
                        else:
                            visitable_neighbours.append(i[0:2])
                return visitable_neighbours

            # StartingPoint
            scr = random.randint(0, ms-1)
            scc = random.randint(0, ms-1)

            # Update starting player position
            self.player.left = scc * SPRITE_SIZE * SCALING_INT
            self.player.bottom = scr * SPRITE_SIZE * SCALING_INT

            ccr, ccc = scr, scc

            # This is the most confusing part of the technique
            map[ccr][ccc] = 'Open'
            finished = False
            while not finished:
                visitable_neighbours = check_neighbours(ccr, ccc)
                if len(visitable_neighbours) != 0:
                    d = random.randint(0, len(visitable_neighbours)-1)
                    ncr, ncc = visitable_neighbours[d]
                    map[ncr][ncc] = 'Open'
                    visited_cells.append([ncr, ncc])
                    ccr, ccc = ncr, ncc
                if len(visitable_neighbours) == 0:
                    try:
                        ccr, ccc = visited_cells.pop()
                        revisited_cells.append([ccr, ccc])
                    except:
                        finished = True

            create()
            end = random.randint(0, len(revisited_cells)) 
            ecr = revisited_cells[end][0] 
            ecc = revisited_cells[end][1] 

            """Important note regarding end and ecr stuff:"""
            # index out of range error when it was at 1 and outside, empty range on inside
            # index out of range here; it's because revisitedcells it's *empty*. no[0][0]. Re-check numbergen
            # also, changing the thing to no longer use revisitedcells will fix it; the new-style end gen.
            # so that'll be fine; don't need revisitedcells once I change that gen technique. corridor_end_cells; don't need to track direction, I think.
            # But yes, this error will be fixed by changing the gen technique for be more fully my own.

            # change end tile position
            self.end_tile.left = ecc * SPRITE_SIZE * SCALING_INT
            self.end_tile.bottom = ecr * SPRITE_SIZE * SCALING_INT
        
        create_maze()

        # Make fog same as the maze
        def create_fog():
            for row in range(MAZE_SIZE):
                    for col in range(MAZE_SIZE):
                        sprite = arcade.Sprite("./images/fog_tile.png", SCALING)

                        x = col * SPRITE_SIZE * SCALING_INT
                        y = row * SPRITE_SIZE * SCALING_INT

                        sprite.left = x
                        sprite.bottom = y
                        self.fog_list.append(sprite)
                        self.all_sprites.append(sprite)
        
        create_fog()

        self.physics_engine = arcade.PhysicsEngineSimple(self.player, self.wall_list) 
        #TODO This does some funny shunting. Maybe consider fixing sprite hitbox? FOr now, change the step size to half.
    
    #Make this call update (turn_update?) with a parameter of 'command'? Only if we want it to be fully turn-based.
    def on_key_press(self, symbol, modifier): #modifier is Shift, ctrl, alt, etc; not used but must be there to work
        if symbol == arcade.key.Q:
            # Quit
            arcade.close_window()

        # Pausing does nothing, essentially; however, this will remain in so that we can use it later if we want
        if symbol == arcade.key.P:
            self.paused = not self.paused

        # top and right chosen because positive in those directions is positive x and y in arcade's positioning
        if symbol == arcade.key.W or symbol == arcade.key.UP:
            self.player.top += SPRITE_SIZE * SCALING_INT // 2 
        if symbol == arcade.key.S or symbol == arcade.key.DOWN:
            self.player.top -= SPRITE_SIZE * SCALING_INT // 2 
        if symbol == arcade.key.A or symbol == arcade.key.LEFT:
            self.player.right -= SPRITE_SIZE * SCALING_INT // 2
        if symbol == arcade.key.D or symbol == arcade.key.RIGHT:
            self.player.right += SPRITE_SIZE * SCALING_INT // 2
        # //2 thing will always come out because sprite size will always be a 2-power
        # However, this is just a temorary fix for the shunting problem cause by sprite being top left, since the
        # hitbox is smaller than 16px; try centering it first or forcing it up to 16px?

    def on_update(self, delta_time: float):
        """
        Update the positions and statuses of all game objects
        If paused, do nothing
    
        Arguments:
            delta_time {float} -- Time since the last update.""" # Currently unused, however, will leave it in in case we want it later
    
        # If paused, don't update anything
        if self.paused:
            return
        
        # End the game if you reach the end
        if self.player.collides_with_sprite(self.end_tile):
            arcade.close_window() 
    
        # Update everything
        self.physics_engine.update() # Easiest thing of my life what the heck; self.all_sprites.update() not needed since we're using a physics engine now
    
        # Keep the player on screen
        if self.player.top > self.height:
            self.player.top = self.height
        if self.player.right > self.width:
            self.player.right = self.width
        if self.player.bottom < 0:
            self.player.bottom = 0
        if self.player.left < 0:
            self.player.left = 0
        
        # State of the art dynamic lighting
        for square in self.fog_list:
            if ((abs(square.center_x - self.player.center_x) + abs(square.center_y - self.player.center_y)) <= 3 * SPRITE_SIZE * SCALING) and (arcade.has_line_of_sight(self.player.position, square.position, self.wall_list)):
                square.alpha = 0
            else:
                square.alpha = 255
    
    def on_draw(self):
        """
        Draw all game objects
        """
        arcade.start_render()
        self.all_sprites.draw()
            
def main():
    """ Main function """
    window = Torchlight(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()

if __name__ == "__main__":
    main()