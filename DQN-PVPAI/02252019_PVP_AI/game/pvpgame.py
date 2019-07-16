import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import math
from scipy.spatial.distance import *
import random

class game:

    #variable#######################################################

    WHITE = (255,255,255)
    w = 1024
    h = 512
    COOLTIME      = 30
    accel_rate    = 0.8
    deccel_rate   = 6
    bullet_speed  = 8
    radius        = 30 # pxel
    rot_rate      = 0.1
    gravitation   = 100
    frame_counter = 0

    DIC_0           = { pygame.K_UP : 0, pygame.K_DOWN:1, pygame.K_LEFT:2, pygame.K_RIGHT:3 }
    DIC_1           = { pygame.K_w : 0, pygame.K_s:1, pygame.K_a:2, pygame.K_d:3 }
    DIC_GUN         = { pygame.K_f: 0, pygame.K_g: 1}
    gun_rate        = [ -rot_rate , rot_rate]         # y,x pos

    #function#######################################################

    def player0(self,x,y):
        global pad, aircraft0
        pad.blit(aircraft0, (x-25,y-25))

    def player1(self,x,y):
        global pad, aircraft1
        pad.blit(aircraft1, (x-25,y-25))

    def gun(self,x,y):
        global pad, gunshot
        pad.blit(gunshot, (x-5,y-5))

    def bullet0(self,x,y):
        global pad, bulletshape1
        pad.blit(bulletshape0, (x-5,y-5))

    def bullet1(self,x,y):
        global pad, bulletshape0
        pad.blit(bulletshape1, (x-5,y-5))

    def init(self):
        global pad, clock, aircraft0, aircraft1, gunshot, bulletshape0, bulletshape1

        pygame.init()
        pad = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('PVP')
        aircraft0 = pygame.image.load('images/airplane0.png')
        aircraft1 = pygame.image.load('images/airplane1.png')
        gunshot = pygame.image.load('images/gun.png')
        bulletshape0 = pygame.image.load('images/bullet0.png')
        bulletshape1 = pygame.image.load('images/bullet1.png')
        clock = pygame.time.Clock()

        self.dist            = [ -self.accel_rate, self.accel_rate ]
        
        self.KEY_PRESSED_0   = [ False, False, False, False ]
        self.MOUSE_PRESSED_0 = False
        self.recent_shot_0   = 0
        self.player_pos_0    = [ self.h / 2, self.w * 0.25 ] # y,x pos
        self.mouse_pos_0     = [ self.h / 2, self.w / 2 ] # y,x pos
        self.gun_pos_0       = [ 0, 0 ]         # y,x pos
        self.pos_change_0    = [ 0, 0 ]
        self.radius_pos_0    = [ 0, 0 ]
        self.bullets_0       = []               # element : [sy, sx, dy, dx]
        
        self.KEY_PRESSED_1   = [ False, False, False, False ]
        self.MOUSE_PRESSED_1 = True
        self.recent_shot_1   = 0
        self.player_pos_1    = [ self.h / 2, self.w * 0.75 ] # y,x pos
        self.mouse_pos_1     = [ self.h / 2, self.w / 2 ]    # y,x pos
        self.pos_change_1    = [ 0, 0 ]
        self.bullets_1       = []                  # element : [sy, sx, dy, dx]
        
        self.gun_rot         = math.pi
        self.KEY_PRESSED_G   = [ False, False ]
        self.gun_pos_1       = [ 0, 0 ]                        # y,x pos
        
        self.gameset = False
        self.winner  = None
        self.theta_r = None
        self.dist_r  = None
        self.poab    = False
        self.theta_m = None
        self.dist_m  = None
        self.gamma_m = None
        
        return self.gameset, self.winner, [self.theta_r, self.dist_r], self.poab, [self.theta_m, self.dist_m, self.gamma_m] 

    def step(self,action_1r,action_1m):
        global pad, clock, aircraft, gunshot, bulletshape

        if not self.gameset:
            
            # key event processing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gameset = True
                if event.type == pygame.KEYDOWN:
                    try:
                        self.KEY_PRESSED_0[self.DIC_0[event.key]] = True
                    except:
                        pass
                    try:
                        self.KEY_PRESSED_1[self.DIC_1[event.key]] = True
                    except:
                        pass
                    try:
                        self.KEY_PRESSED_G[self.DIC_GUN[event.key]] = True
                    except:
                        pass
                    if event.key == pygame.K_SPACE:
                        self.MOUSE_PRESSED_1 = True

                if event.type == pygame.KEYUP:
                    try:
                        self.KEY_PRESSED_0[self.DIC_0[event.key]] = False
                    except:
                        pass
                    try:
                        self.KEY_PRESSED_1[self.DIC_1[event.key]] = False
                    except:
                        pass
                    try:
                        self.KEY_PRESSED_G[self.DIC_GUN[event.key]] = False
                    except:
                        pass
                    if event.key == pygame.K_SPACE:
                        self.MOUSE_PRESSED_1 = False
                
                if event.type == pygame.MOUSEMOTION:
                    self.mouse_pos_0 = [event.pos[1], event.pos[0]]

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.MOUSE_PRESSED_0 = True
                
                if event.type == pygame.MOUSEBUTTONUP:
                    self.MOUSE_PRESSED_0 = False
            

            # player1 rotate action
            for i in range(2):
                if action_1r == i:
                    self.KEY_PRESSED_G[i] = True
                else:
                    self.KEY_PRESSED_G[i] = False

            #player1 move action
            for i in range(2):
                if action_1m//2 == i:
                    self.KEY_PRESSED_1[i] = True
                else:
                    self.KEY_PRESSED_1[i] = False

            for i in range(2):
                if action_1m%2 == i:
                    self.KEY_PRESSED_1[2+i] = True
                else:
                    self.KEY_PRESSED_1[2+i] = False

            # player movement
            for i in range(4):
                if self.KEY_PRESSED_0[i]:
                    self.pos_change_0[int(i/2)] += self.dist[i%2]
                
                if self.KEY_PRESSED_1[i]:
                    self.pos_change_1[int(i/2)] += self.dist[i%2]
                    
            for i in range(2):
                self.player_pos_0[i] += self.pos_change_0[i]
                self.player_pos_0[i]  = max(0, min(self.player_pos_0[i], self.h if i%2==0 else self.w))
                self.pos_change_0[i] /= (self.deccel_rate+1)/self.deccel_rate
                
                self.player_pos_1[i] += self.pos_change_1[i]
                self.player_pos_1[i]  = max(0, min(self.player_pos_1[i], self.h if i%2==0 else self.w))
                self.pos_change_1[i] /= (self.deccel_rate+1)/self.deccel_rate

            # gun movement
            disp          = [self.mouse_pos_0[0]-self.player_pos_0[0], self.mouse_pos_0[1]-self.player_pos_0[1]]
            hypo          = math.sqrt(disp[0]*disp[0]+disp[1]*disp[1])
            self.radius_pos_0  = [self.radius*disp[0]/hypo, self.radius*disp[1]/hypo]
            self.gun_pos_0     = [self.player_pos_0[0]+self.radius_pos_0[0],self.player_pos_0[1]+self.radius_pos_0[1]]
            
            for i in range(2):
                if self.KEY_PRESSED_G[i] == True:
                    self.gun_rot+=self.gun_rate[i]
            self.gun_pos_1     = [self.player_pos_1[0]+math.sin(self.gun_rot)*self.radius,self.player_pos_1[1]+math.cos(self.gun_rot)*self.radius]

            # bullet add
            if self.MOUSE_PRESSED_0 and self.frame_counter - self.recent_shot_0 > self.COOLTIME:
                self.recent_shot_0 = self.frame_counter
                mx, my = self.gun_pos_0[1], self.gun_pos_0[0]
                sx = self.player_pos_0[1]
                sy = self.player_pos_0[0]
                
                b_disp = [my-sy, mx-sx]
                b_hypo  = math.sqrt(b_disp[0]*b_disp[0]+b_disp[1]*b_disp[1])
                
                dy = self.bullet_speed*b_disp[0]/b_hypo
                dx = self.bullet_speed*b_disp[1]/b_hypo

                self.bullets_0.append([my,mx,dy,dx])
            
            if self.MOUSE_PRESSED_1 and self.frame_counter - self.recent_shot_1 > self.COOLTIME:
                self.recent_shot_1 = self.frame_counter
                mx, my = self.gun_pos_1[1], self.gun_pos_1[0]
                sx = self.player_pos_1[1]
                sy = self.player_pos_1[0]
                        
                b_disp = [my-sy, mx-sx]
                b_hypo  = math.sqrt(b_disp[0]*b_disp[0]+b_disp[1]*b_disp[1])
                        
                dy = self.bullet_speed*b_disp[0]/b_hypo
                dx = self.bullet_speed*b_disp[1]/b_hypo

                self.bullets_1.append([my,mx,dy,dx])
                
            # bullet movement and collision and player gravitation
            i = 0
            size = len(self.bullets_0)
            while i < size:
                if (self.bullets_0[i][0]< -0.2 * self.h or self.bullets_0[i][0] > 1.2 * self.h or self.bullets_0[i][1]< -0.2 * self.w or self.bullets_0[i][1] > 1.2 * self.w):
                    del self.bullets_0[i]
                    size-=1
                elif cdist([[ self.bullets_0[i][0], self.bullets_0[i][1] ]],[self.player_pos_1]) < 25:
                    self.winner  = 0
                    self.gameset = True
                    break
                else : 
                    # dy = self.player_pos_1[0] - self.bullets_0[i][0]
                    # dx = self.player_pos_1[1] - self.bullets_0[i][1]
                    # hy = math.sqrt(dx*dx+dy*+dy)
                    # self.bullets_0[i][2] += (self.player_pos_1[0] - self.bullets_0[i][0]) / math.pow(hy,2.5)  * self.gravitation
                    # self.bullets_0[i][3] += (self.player_pos_1[1] - self.bullets_0[i][1]) / math.pow(hy,2.5)  * self.gravitation
                    
                    self.bullets_0[i][0] += self.bullets_0[i][2]
                    self.bullets_0[i][1] += self.bullets_0[i][3]
                    i+=1

            i = 0
            size = len(self.bullets_1)
            while i < size:
                if (self.bullets_1[i][0] < -0.2 * self.h or self.bullets_1[i][0] > 1.2 * self.h or self.bullets_1[i][1]< -0.2 * self.w or self.bullets_1[i][1] > 1.2 * self.w):
                    del self.bullets_1[i]
                    size-=1
                elif cdist([[ self.bullets_1[i][0], self.bullets_1[i][1] ]],[self.player_pos_0]) < 25:
                    self.winner  = 1
                    self.gameset = True
                    break
                else : 
                    # dy = self.player_pos_0[0] - self.bullets_1[i][0]
                    # dx = self.player_pos_0[1] - self.bullets_1[i][1]
                    # hy = math.sqrt(dx*dx+dy*+dy)
                    # self.bullets_1[i][2] += (self.player_pos_0[0] - self.bullets_1[i][0]) / math.pow(hy,2.5)  * self.gravitation
                    # self.bullets_1[i][3] += (self.player_pos_0[1] - self.bullets_1[i][1]) / math.pow(hy,2.5)  * self.gravitation
                    
                    self.bullets_1[i][0] += self.bullets_1[i][2]
                    self.bullets_1[i][1] += self.bullets_1[i][3]
                    i+=1
            
            # background color
            pad.fill(self.WHITE)

            # draw image
            self.player0(self.player_pos_0[1],self.player_pos_0[0])
            self.gun(self.gun_pos_0[1],self.gun_pos_0[0])
            for bul in self.bullets_0:
                self.bullet0(bul[1],bul[0])

            self.player1(self.player_pos_1[1],self.player_pos_1[0])
            self.gun(self.gun_pos_1[1],self.gun_pos_1[0])
            for bul in self.bullets_1:
                self.bullet1(bul[1],bul[0])

            # update
            pygame.display.update()
            self.frame_counter+=1

    
        # return values
        # gameset, winner, enemy's posistion, presence or absence of bullet, proximate bullet's position, proximate bullet's move angle


        dy = self.player_pos_0[0]-self.player_pos_1[0]
        dx = self.player_pos_0[1]-self.player_pos_1[1]
        self.theta_r = math.atan2(dx,dy)
        self.dist_r  = cdist([self.player_pos_0],[self.player_pos_1])[0]
        
        self.poab = len(self.bullets_0)>0
        mindist = 100000000
        for bullet in self.bullets_0:
            distmp = cdist([[ bullet[0], bullet[1] ]],[self.player_pos_1])
            if mindist > distmp:
                mindist = distmp
                dy = bullet[0]-self.player_pos_1[0]
                dx = bullet[1]-self.player_pos_1[1]
                self.theta_m = math.atan2(dx,dy)
                self.dist_m  = cdist([[ bullet[0], bullet[1] ]],[self.player_pos_1])[0]
                self.gamma_m = math.atan2(bullet[3],bullet[2])

        clock.tick(60)

        return self.gameset, self.winner, [self.theta_r, self.dist_r], self.poab, [self.theta_m, self.dist_m, self.gamma_m] 