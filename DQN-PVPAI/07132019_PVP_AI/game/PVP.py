import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import math
from math import *
from scipy.spatial.distance import *

# AI VS AI module
class game:
    def __init__(self):
        # const value
        self.WHITE         = (255,255,255)
        self.w             = 1024
        self.h             = 512
        self.cooltime      = 35
        self.accel         = 0.8
        self.deccel        = 6
        self.rotation_rate = 0.05
        self.speed         = 7
        self.radius        = 30 # pxel
        self.rot_rate      = 0.1

        self.reset()

    def pygame_init(self):
        # config
        #pygame.init()
        pygame.display.set_caption('PVP')

        # image config
        self.pad           = pygame.display.set_mode((self.w, self.h))
        self.aircraft0     = pygame.image.load('images/airplane0.png')
        self.aircraft1     = pygame.image.load('images/airplane1.png')
        self.gunshot       = pygame.image.load('images/gun.png')
        self.bulletshape0  = pygame.image.load('images/bullet0.png')
        self.bulletshape1  = pygame.image.load('images/bullet1.png')
        self.clock         = pygame.time.Clock() 

    def reset(self):
        self.pygame_init()
        
        # player status values
        self.FRAME_COUNTER = 0 # frame counter
        self.rotation      = [ -self.rotation_rate, self.rotation_rate]

        self.PLAYER0       = [ self.h / 2, self.w * 0.25 ]
        self.PLAYER1       = [ self.h / 2, self.w * 0.75 ]
        
        self.CURSOR0       = [ self.h / 2, self.w * 0.25 ]
        self.CURSOR1       = [ self.h / 2, self.w * 0.75 ]

        self.GUN0          = [ None, None ]
        self.GUN1          = [ None, None ]
        self.GUN_ROT0      = 0
        self.GUN_ROT1      = math.pi

        self.x = 0
        self.y = 0
        self.ROTATE0       = [ False, False ]
        self.MOVE0         = [ False, False, False, False ]
        self.ROTATE1       = [ False, False ]
        self.MOVE1         = [ False, False, False, False ]
        self.DISPL0        = [ 0, 0 ]
        self.DISPL1        = [ 0, 0 ]
        
        self.BULLETS0      = [  ]
        self.BULLETS1      = [  ]
        self.RECENT_SHOT0  = 0
        self.RECENT_SHOT1  = 0
        
        self.displacement  = [ -self.accel, self.accel ]
        self.done          = False
        self.winner        = [None]
        self.sm0           = [None]
        self.sr0           = [None]
        self.sm1           = [None]
        self.sr1           = [None]

        
        self.step(0, 0, 0, 0)
        
        return self.done, self.winner, self.sm0, self.sm1, self.sm1, self.sm0

    def update_status_move(self):
        #status_rotate
        self.sm0 = self.PLAYER0[:]
        self.sm1 = self.PLAYER1[:]

        for bullet in self.BULLETS1:
            self.sm0.append(bullet[0])
            self.sm0.append(bullet[1])

        for bullet in self.BULLETS0:
            self.sm1.append(bullet[0])
            self.sm1.append(bullet[1])

        for i in range(10-len(self.BULLETS1)):
            self.sm0.append(0)
            self.sm0.append(0)
    
        for i in range(10-len(self.BULLETS0)):
            self.sm1.append(0)
            self.sm1.append(0)

        return self.sm0, self.sm1

    def step(self, ar0, am0, ar1, am1):
        if not self.done:
            # action process
            for i in range(2):
                self.MOVE0[ i ] =  True if am0 %3 == i else False
                self.MOVE0[2+i] =  True if am0//3 == i else False
                self.MOVE1[ i ] =  True if am1 %3 == i else False
                self.MOVE1[2+i] =  True if am1//3 == i else False
            
            # player movement process
            for i in range(4):
                self.DISPL0[i//2] += self.displacement[i%2] if self.MOVE0[i] == True else 0
                self.DISPL1[i//2] += self.displacement[i%2] if self.MOVE1[i] == True else 0
            
            for i in range(2):
                self.PLAYER0[i] += self.DISPL0[i]
                self.PLAYER0[i]  = max(0, min(self.PLAYER0[i], self.h if i%2==0 else self.w))
                self.DISPL0[i] /= ( self.deccel + 1 ) / self.deccel
                
                self.PLAYER1[i] += self.DISPL1[i]
                self.PLAYER1[i]  = max(0, min(self.PLAYER1[i], self.h if i%2==0 else self.w))
                self.DISPL1[i] /= ( self.deccel + 1 ) / self.deccel
            
            # gun movement process
            print(ar0,ar1)
            self.GUN_ROT0 += ar0
            self.GUN_ROT1 += ar1
            
            self.GUN0 = [ self.PLAYER0[0]+math.sin(self.GUN_ROT0)*self.radius, self.PLAYER0[1]+math.cos(self.GUN_ROT0)*self.radius ]
            self.GUN1 = [ self.PLAYER1[0]+math.sin(self.GUN_ROT1)*self.radius, self.PLAYER1[1]+math.cos(self.GUN_ROT1)*self.radius ]
            
            # add bullet
            if self.FRAME_COUNTER - self.RECENT_SHOT0 > self.cooltime:
                self.RECENT_SHOT0 = self.FRAME_COUNTER
                mx, my = self.GUN0[1], self.GUN0[0]
                sx, sy = self.PLAYER0[1], self.PLAYER0[0]
                dx, dy = mx - sx, my - sy
                hypo = math.sqrt(dx*dx+dy*dy)
                px, py = self.speed * dx/hypo, self.speed * dy/hypo

                self.BULLETS0.append([my, mx, py, px])
            
            if self.FRAME_COUNTER - self.RECENT_SHOT1 > self.cooltime:
                self.RECENT_SHOT1 = self.FRAME_COUNTER
                mx, my = self.GUN1[1], self.GUN1[0]
                sx, sy = self.PLAYER1[1], self.PLAYER1[0]
                dx, dy = mx - sx, my - sy
                hypo = math.sqrt(dx*dx+dy*dy)
                px, py = self.speed * dx/hypo, self.speed * dy/hypo

                self.BULLETS1.append([my, mx, py, px])
                
            # bullet movement process
            i = 0
            size = len(self.BULLETS0)
            while i < size:
                if (self.BULLETS0[i][0]< -0.2 * self.h or self.BULLETS0[i][0] > 1.2 * self.h or self.BULLETS0[i][1]< -0.2 * self.w or self.BULLETS0[i][1] > 1.2 * self.w):
                    del self.BULLETS0[i]
                    size-=1
                elif cdist([[ self.BULLETS0[i][0], self.BULLETS0[i][1] ]],[self.PLAYER1]) < 25:
                    self.winner  = 0
                    self.done = True
                    break
                else : 
                    self.BULLETS0[i][0] += self.BULLETS0[i][2]
                    self.BULLETS0[i][1] += self.BULLETS0[i][3]
                    i+=1
            i = 0
            size = len(self.BULLETS1)
            while i < size:
                if (self.BULLETS1[i][0]< -0.2 * self.h or self.BULLETS1[i][0] > 1.2 * self.h or self.BULLETS1[i][1]< -0.2 * self.w or self.BULLETS1[i][1] > 1.2 * self.w):
                    del self.BULLETS1[i]
                    size-=1
                elif cdist([[ self.BULLETS1[i][0], self.BULLETS1[i][1] ]],[self.PLAYER0]) < 25:
                    self.winner  = 1
                    self.done = True
                    break
                else : 
                    self.BULLETS1[i][0] += self.BULLETS1[i][2]
                    self.BULLETS1[i][1] += self.BULLETS1[i][3]
                    i+=1
        return self.done, self.winner, self.sm0, self.sm1, self.sm1, self.sm0

    def render(self):

        self.pad.fill(self.WHITE)
        self.draw_player0(self.PLAYER0[1],self.PLAYER0[0])
        self.draw_gun(self.GUN0[1],self.GUN0[0])
        for bul in self.BULLETS0:
            self.draw_bullet0(bul[1],bul[0])

        self.draw_player1(self.PLAYER1[1],self.PLAYER1[0])
        self.draw_gun(self.GUN1[1],self.GUN1[0])
        for bul in self.BULLETS1:
            self.draw_bullet1(bul[1],bul[0])

        pygame.display.update()
        pygame.event.get()
        self.clock.tick(100)
        
    def draw_player0(self,x,y):
        self.pad.blit(self.aircraft0, (x-25,y-25))

    def draw_player1(self,x,y):
        self.pad.blit(self.aircraft1, (x-25,y-25))

    def draw_gun(self,x,y):
        self.pad.blit(self.gunshot, (x-5,y-5))

    def draw_bullet0(self,x,y):
        self.pad.blit(self.bulletshape0, (x-5,y-5))

    def draw_bullet1(self,x,y):
        self.pad.blit(self.bulletshape1, (x-5,y-5))

    def player_action(self):
        dic0 = [pygame.K_w, pygame.K_s]
        dic1 = [pygame.K_a, pygame.K_d]

        DIC0 = { pygame.K_w : 0, pygame.K_s : 1 }
        DIC1 = { pygame.K_a : 0, pygame.K_d : 1 }

        self.KPRESSED0 = [False, False]
        self.KPRESSED1 = [False, False]
        
        m0, m1 = 2, 2
        diff = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameset = True
            if event.type == pygame.MOUSEMOTION:
                y = event.pos[1]
                x = event.pos[0]

                diff = atan2((y-self.PLAYER0[0]),(x-self.PLAYER0[1])) - atan2((self.y-self.PLAYER0[0]),(self.x-self.PLAYER0[1]))
                # diff *= 180 / math.pi
                
                self.y = event.pos[1]
                self.x = event.pos[0]
            
        key = pygame.key.get_pressed()
        for k in range(2):
            if key[dic0[k]] == True:
                self.KPRESSED0[k] = True

        for k in range(2):
            if key[dic1[k]] == True:
                self.KPRESSED1[k] = True

        for i in range(2):
            if self.KPRESSED1[i] == True:
                m0 = i
            if self.KPRESSED0[i] == True:
                m1 = i

        return m0, m1, diff
        

