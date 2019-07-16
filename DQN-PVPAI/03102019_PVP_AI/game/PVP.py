import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import math
from scipy.spatial.distance import *
import numpy as np

# AI VS AI module
class game:
    def __init__(self):
        # const value
        self.WHITE         = (255,255,255)
        self.w             = 1024
        self.h             = 512
        self.cooltime      = 50
        self.accel         = 1
        self.deccel        = 6
        self.rotation_rate = 0.1
        self.speed         = 5
        self.radius        = 30 # pxel
        self.rot_rate      = 0.05

        self.reset()

    def pygame_init(self):
        # config
        pygame.init()
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
        # self.pygame_init()
        
        # player status values
        self.FRAME_COUNTER = 0 # frame counter
        self.rotation      = [ -self.rotation_rate, self.rotation_rate]

        self.PLAYER0       = [ self.h / 2, self.w * 0.25 ]
        self.PLAYER1       = [ self.h / 2, self.w * 0.75 ]
        
        self.PLAYER0_tmp   = self.PLAYER0
        self.PLAYER1_tmp   = self.PLAYER1

        self.GUN0          = [ None, None ]
        self.GUN1          = [ None, None ]
        self.GUN_ROT0      = 0
        self.GUN_ROT1      = math.pi

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
        self.sr0           = [None, None]
        self.sm1           = [None]
        self.sr1           = [None, None]

        self.fsr0          = [None]
        self.fsr1          = [None]
        self.fsr0_capture  = []
        self.fsr1_capture  = []

        self.step(0,0,0,0)


        return self.done, self.winner, self.sm0, self.sm1, self.sr0, self.sr1

    def update_status_rotate(self):
            #status_rotate
        dist = cdist([self.PLAYER0],[self.PLAYER1])
        dy, dx = self.PLAYER0[0]-self.PLAYER1[0] , self.PLAYER0[1]-self.PLAYER1[1]

        dist0 = cdist([self.PLAYER0],[self.GUN0])
        dy0, dx0 = self.GUN0[0]-self.PLAYER0[0] , self.GUN0[1]-self.PLAYER0[1]
        
        dist1 = cdist([self.PLAYER1],[self.GUN1])
        dy1, dx1 = self.GUN1[0]-self.PLAYER1[0] , self.GUN1[1]-self.PLAYER1[1]

        a0,  a1  = math.atan2(-dx, -dy), math.atan2(dx , dy )
        da0, da1 = math.atan2(dx0, dy0), math.atan2(dx1, dy1)
        
        theta0, theta1 = a0 - da0, a1 - da1

        if math.pi*2-math.fabs(theta0) < math.fabs(theta0):
            theta0 = (math.pi * 2 -math.fabs(theta0)) * -1 * theta0 / math.fabs(theta0) 
            
        if math.pi*2-math.fabs(theta1) < math.fabs(theta1):
            theta1 = (math.pi * 2 -math.fabs(theta1)) * -1 * theta1 / math.fabs(theta1) 

        rx0, ry0 = self.DISPL0[1], self.DISPL0[0] 
        rx1, ry1 = self.DISPL1[1], self.DISPL1[0] 

        self.sr0 = [dist, theta0, rx1, ry1]
        self.sr1 = [dist, theta1, rx0, ry0]
        
        return self.sr0, self.sr1
    
    
    # def update_status_move(self):
    #     #status_rotate
    #     xc0, yc0 = self.w/2 - self.PLAYER0[1], self.w/2 - self.PLAYER0[0]
    #     xc1, yc1 = self.w/2 - self.PLAYER1[1], self.w/2 - self.PLAYER1[0]
    #     xm0, ym0, xm1, ym1 = None, None, None, None
    #     gamma0, gamma1 = None, None

    #     tmpdist0 = []
    #     tmpdist1 = []

    #     tmpbul0 = []
    #     tmpbul1 = []

    #     self.sm0 = [[self.w // 2 - self.PLAYER0[1], self.h // 2 - self.PLAYER0[0]], self.PLAYER1]
    #     self.sm1 = [[self.w // 2 - self.PLAYER1[1], self.h // 2 - self.PLAYER1[0]]]
        
        
    #     for bullet in self.BULLETS0:
    #         tmpdist0.append(cdist([[ bullet[0], bullet[1] ]],[self.PLAYER1]))
            
    #     for bullet in self.BULLETS1:
    #         tmpdist1.append(cdist([[ bullet[0], bullet[1] ]],[self.PLAYER0]))
        
    #     for i in range(min(2,len(self.BULLETS0))):
    #         index = np.argmin(tmpdist0)
    #         tmpbul0.append(self.BULLETS0[index])
    #         tmpdist0[index] = 10000000

    #     for i in range(min(2,len(self.BULLETS1))):
    #         index = np.argmin(tmpdist1)
    #         tmpbul1.append(self.BULLETS1[index])
    #         tmpdist1[index] = 10000000
        
    #     for bullet in tmpbul0:
    #         ym = bullet[0]-self.PLAYER1[0]
    #         xm = bullet[1]-self.PLAYER1[1]
    #         self.sm0 = np.append(self.sm0,xm)
    #         self.sm0 = np.append(self.sm0,ym)
    #         self.sm0 = np.append(self.sm0,bullet[3])
    #         self.sm0 = np.append(self.sm0,bullet[2])
        
    #     for bullet in tmpbul1:
    #         ym = bullet[0]-self.PLAYER0[0]
    #         xm = bullet[1]-self.PLAYER0[1]
    #         gamma = math.atan2(bullet[3],bullet[2])
    #         self.sm1 = np.append(self.sm1,xm)
    #         self.sm1 = np.append(self.sm1,ym)
    #         self.sm1 = np.append(self.sm1,bullet[3])
    #         self.sm1 = np.append(self.sm1,bullet[2])

    #     return self.sm0, self.sm1
    def update_status_move(self):
        #status_move
        self.sm0 = self.PLAYER0[:]
        self.sm1 = self.PLAYER1[:]

        for bullet in self.BULLETS1:
            self.sm0.append(bullet[0])
            self.sm0.append(bullet[1])
            self.sm0.append(bullet[2])
            self.sm0.append(bullet[3])
            
        for i in range(10-len(self.BULLETS1)):
            self.sm0.append(0)
            self.sm0.append(0)
            self.sm0.append(0)
            self.sm0.append(0)

        for bullet in self.BULLETS0:
            self.sm1.append(bullet[0])
            self.sm1.append(bullet[1])
            self.sm1.append(bullet[2])
            self.sm1.append(bullet[3])

        for i in range(10-len(self.BULLETS0)):
            self.sm1.append(0)
            self.sm1.append(0)
            self.sm1.append(0)
            self.sm1.append(0)

        
    def step(self, ar0, am0, ar1, am1):
        if not self.done:
            # action process
            for i in range(2):
                self.MOVE0[ i ] =  True if am0 %3 == i else False
                self.MOVE0[2+i] =  True if am0//3 == i else False
                self.MOVE1[ i ] =  True if am1 %3 == i else False
                self.MOVE1[2+i] =  True if am1//3 == i else False
                
                self.ROTATE0[i]  = True if i == ar0 else False
                self.ROTATE1[i]  = True if i == ar1 else False
            
            # player movement process
            for i in range(4):
                self.DISPL0[i//2] += self.displacement[i%2] if self.MOVE0[i] == True else 0
                self.DISPL1[i//2] += self.displacement[i%2] if self.MOVE1[i] == True else 0
            
            self.PLAYER0_tmp = self.PLAYER0
            self.PLAYER1_tmp = self.PLAYER1
            
            for i in range(2):
                self.PLAYER0[i] += self.DISPL0[i]
                self.PLAYER0[i]  = max(0, min(self.PLAYER0[i], self.h if i%2==0 else self.w))
                self.DISPL0[i] /= ( self.deccel + 1 ) / self.deccel
                
                self.PLAYER1[i] += self.DISPL1[i]
                self.PLAYER1[i]  = max(0, min(self.PLAYER1[i], self.h if i%2==0 else self.w))
                self.DISPL1[i] /= ( self.deccel + 1 ) / self.deccel
            
            # gun movement process
            for i in range(2):
                self.GUN_ROT0 += self.rotation[i] if self.ROTATE0[i] == True else 0
                self.GUN_ROT1 += self.rotation[i] if self.ROTATE1[i] == True else 0
            
            self.GUN0 = [ self.PLAYER0[0]+math.sin(self.GUN_ROT0)*self.radius, self.PLAYER0[1]+math.cos(self.GUN_ROT0)*self.radius ]
            self.GUN1 = [ self.PLAYER1[0]+math.sin(self.GUN_ROT1)*self.radius, self.PLAYER1[1]+math.cos(self.GUN_ROT1)*self.radius ]
            
            bsr0, bsr1 = self.sr0, self.sr1
            asr0, asr1 = self.update_status_rotate()

            # add bullet
            if self.FRAME_COUNTER - self.RECENT_SHOT0 > self.cooltime:
                self.RECENT_SHOT0 = self.FRAME_COUNTER
                mx, my = self.GUN0[1], self.GUN0[0]
                sx, sy = self.PLAYER0[1], self.PLAYER0[0]
                dx, dy = mx - sx, my - sy
                hypo = math.sqrt(dx*dx+dy*dy)
                px, py = self.speed * dx/hypo, self.speed * dy/hypo
                self.BULLETS0.append([my, mx, py, px])

                self.fsr0_capture.append([bsr0, ar0, asr0])

            
            if self.FRAME_COUNTER - self.RECENT_SHOT1 > self.cooltime:
                self.RECENT_SHOT1 = self.FRAME_COUNTER
                mx, my = self.GUN1[1], self.GUN1[0]
                sx, sy = self.PLAYER1[1], self.PLAYER1[0]
                dx, dy = mx - sx, my - sy
                hypo = math.sqrt(dx*dx+dy*dy)
                px, py = self.speed * dx/hypo, self.speed * dy/hypo
                self.BULLETS1.append([my, mx, py, px])
                
                self.fsr1_capture.append([bsr1, ar1, asr1])

                
            # bullet movement process
            i = 0
            size = len(self.BULLETS0)
            while i < size:
                if (self.BULLETS0[i][0]< -0.2 * self.h or self.BULLETS0[i][0] > 1.2 * self.h or self.BULLETS0[i][1]< -0.2 * self.w or self.BULLETS0[i][1] > 1.2 * self.w):
                    del self.BULLETS0[i]
                    del self.fsr0_capture[i]
                    size-=1
                elif cdist([[ self.BULLETS0[i][0], self.BULLETS0[i][1] ]],[self.PLAYER1]) < 25:
                    self.winner  = 0
                    self.done = True
                    self.fsr0 = self.fsr0_capture[i]
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
                    del self.fsr1_capture[i]
                    size-=1
                elif cdist([[ self.BULLETS1[i][0], self.BULLETS1[i][1] ]],[self.PLAYER0]) < 25:
                    self.winner  = 1
                    self.done = True
                    self.fsr1 = self.fsr1_capture[i]
                    break
                else : 
                    self.BULLETS1[i][0] += self.BULLETS1[i][2]
                    self.BULLETS1[i][1] += self.BULLETS1[i][3]
                    i+=1

            # display config
            self.FRAME_COUNTER+=1
            # self.render()

        self.update_status_move()
        
        return self.done, self.winner, self.sm0, self.sm1, self.sr0, self.sr1

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
        self.clock.tick(200)
        
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameset = True
            if event.type == pygame.MOUSEMOTION:
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

        return m0, m1, self.x, self.y
        

