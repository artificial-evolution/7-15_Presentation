import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import math
from scipy.spatial.distance import *
import time

WHITE = (255,255,255)
w = 1024
h = 512

def player0(x,y):
    global pad, aircraft0
    pad.blit(aircraft0, (x-25,y-25))

def player1(x,y):
    global pad, aircraft1
    pad.blit(aircraft1, (x-25,y-25))

def gun(x,y):
    global pad, gunshot
    pad.blit(gunshot, (x-5,y-5))

def bullet0(x,y):
    global pad, bulletshape1
    pad.blit(bulletshape0, (x-5,y-5))

def bullet1(x,y):
    global pad, bulletshape0
    pad.blit(bulletshape1, (x-5,y-5))

def run():
    global pad, clock, aircraft, gunshot, bulletshape
    COOLTIME     = 0.2
    accel_rate   = 0.8
    deccel_rate  = 6
    bullet_speed = 8
    radius       = 30 # pxel
    rot_rate     = 0.1
    gravitation  = 100
    crashed = False

    dist            = [ -accel_rate, accel_rate ]
    
    DIC_0           = { pygame.K_UP : 0, pygame.K_DOWN:1, pygame.K_LEFT:2, pygame.K_RIGHT:3 }
    KEY_PRESSED_0   = [ False, False, False, False ]
    MOUSE_PRESSED_0 = False
    recent_shot_0   = time.time()
    player_pos_0    = [ h / 2, w * 0.25 ] # y,x pos
    mouse_pos_0     = [ h / 2, w / 2 ] # y,x pos
    gun_pos_0       = [ 0, 0 ]         # y,x pos
    pos_change_0    = [ 0, 0 ]
    radius_pos_0    = [ 0, 0 ]
    bullets_0       = []               # element : [sy, sx, dy, dx]
    
    DIC_1           = { pygame.K_w : 0, pygame.K_s:1, pygame.K_a:2, pygame.K_d:3 }
    KEY_PRESSED_1   = [ False, False, False, False ]
    MOUSE_PRESSED_1 = False
    recent_shot_1   = time.time()
    player_pos_1    = [ h / 2, w * 0.75 ] # y,x pos
    mouse_pos_1     = [ h / 2, w / 2 ]    # y,x pos
    pos_change_1    = [ 0, 0 ]
    bullets_1       = []                  # element : [sy, sx, dy, dx]
    
    gun_rot         = 0
    DIC_GUN         = { pygame.K_f: 0, pygame.K_g: 1}
    gun_rate        = [ -rot_rate , rot_rate]         # y,x pos
    KEY_PRESSED_G   = [ False, False ]
    gun_pos_1       = [ 0, 0 ]                        # y,x pos

    while not crashed:
        # key event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                try:
                    KEY_PRESSED_0[DIC_0[event.key]] = True
                except:
                    pass
                try:
                    KEY_PRESSED_1[DIC_1[event.key]] = True
                except:
                    pass
                try:
                    KEY_PRESSED_G[DIC_GUN[event.key]] = True
                except:
                    pass
                if event.key == pygame.K_SPACE:
                    MOUSE_PRESSED_1 = True

            if event.type == pygame.KEYUP:
                try:
                    KEY_PRESSED_0[DIC_0[event.key]] = False
                except:
                    pass
                try:
                    KEY_PRESSED_1[DIC_1[event.key]] = False
                except:
                    pass
                try:
                    KEY_PRESSED_G[DIC_GUN[event.key]] = False
                except:
                    pass
                if event.key == pygame.K_SPACE:
                    MOUSE_PRESSED_1 = False
            
            if event.type == pygame.MOUSEMOTION:
                mouse_pos_0 = [event.pos[1], event.pos[0]]

            if event.type == pygame.MOUSEBUTTONDOWN:
                MOUSE_PRESSED_0 = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                MOUSE_PRESSED_0 = False

        # player movement
        for i in range(4):
            if KEY_PRESSED_0[i]:
                pos_change_0[int(i/2)] += dist[i%2]
            
            if KEY_PRESSED_1[i]:
                pos_change_1[int(i/2)] += dist[i%2]
                
        for i in range(2):
            player_pos_0[i] += pos_change_0[i]
            player_pos_0[i]  = max(0, min(player_pos_0[i], h if i%2==0 else w))
            pos_change_0[i] /= (deccel_rate+1)/deccel_rate
            
            player_pos_1[i] += pos_change_1[i]
            player_pos_1[i]  = max(0, min(player_pos_1[i], h if i%2==0 else w))
            pos_change_1[i] /= (deccel_rate+1)/deccel_rate

        # gun movement
        disp          = [mouse_pos_0[0]-player_pos_0[0], mouse_pos_0[1]-player_pos_0[1]]
        hypo          = math.sqrt(disp[0]*disp[0]+disp[1]*disp[1])
        radius_pos_0  = [radius*disp[0]/hypo, radius*disp[1]/hypo]
        gun_pos_0     = [player_pos_0[0]+radius_pos_0[0],player_pos_0[1]+radius_pos_0[1]]
        
        for i in range(2):
            if KEY_PRESSED_G[i] == True:
                gun_rot+=gun_rate[i]
        gun_pos_1     = [player_pos_1[0]+math.sin(gun_rot)*radius,player_pos_1[1]+math.cos(gun_rot)*radius]

        # bullet add
        if MOUSE_PRESSED_0 and time.time() - recent_shot_0 > COOLTIME:
            recent_shot_0 = time.time()
            mx, my = gun_pos_0[1], gun_pos_0[0]
            sx = player_pos_0[1]
            sy = player_pos_0[0]
            
            b_disp = [my-sy, mx-sx]
            b_hypo  = math.sqrt(b_disp[0]*b_disp[0]+b_disp[1]*b_disp[1])
            
            dy = bullet_speed*b_disp[0]/b_hypo
            dx = bullet_speed*b_disp[1]/b_hypo

            bullets_0.append([my,mx,dy,dx])
        
        if MOUSE_PRESSED_1 and time.time() - recent_shot_1 > COOLTIME:
            recent_shot_1 = time.time()
            mx, my = gun_pos_1[1], gun_pos_1[0]
            sx = player_pos_1[1]
            sy = player_pos_1[0]
                    
            b_disp = [my-sy, mx-sx]
            b_hypo  = math.sqrt(b_disp[0]*b_disp[0]+b_disp[1]*b_disp[1])
                    
            dy = bullet_speed*b_disp[0]/b_hypo
            dx = bullet_speed*b_disp[1]/b_hypo

            bullets_1.append([my,mx,dy,dx])
            
        # bullet movement and collision and player gravitation
        i = 0
        size = len(bullets_0)
        while i < size:
            if (bullets_0[i][0]< -0.2 * h or bullets_0[i][0] > 1.2 * h or bullets_0[i][1]< -0.2 * w or bullets_0[i][1] > 1.2 * w):
                del bullets_0[i]
                size-=1
            elif cdist([[ bullets_0[i][0], bullets_0[i][1] ]],[player_pos_1]) < 25:
                print("Red win")
                crashed = True
                break
            else : 
                dy = player_pos_1[0] - bullets_0[i][0]
                dx = player_pos_1[1] - bullets_0[i][1]
                hy = math.sqrt(dx*dx+dy*+dy)
                bullets_0[i][2] += (player_pos_1[0] - bullets_0[i][0]) / math.pow(hy,2.5)  * gravitation
                bullets_0[i][3] += (player_pos_1[1] - bullets_0[i][1]) / math.pow(hy,2.5)  * gravitation
                
                bullets_0[i][0] += bullets_0[i][2]
                bullets_0[i][1] += bullets_0[i][3]
                i+=1

        i = 0
        size = len(bullets_1)
        while i < size:
            if (bullets_1[i][0]<0 or bullets_1[i][0] > h or bullets_1[i][1]<0 or bullets_1[i][1] > w):
                del bullets_1[i]
                size-=1
            elif cdist([[ bullets_1[i][0], bullets_1[i][1] ]],[player_pos_0]) < 25:
                print("Blue win")
                crashed = True
                break
            else : 
                dy = player_pos_0[0] - bullets_1[i][0]
                dx = player_pos_0[1] - bullets_1[i][1]
                hy = math.sqrt(dx*dx+dy*+dy)
                bullets_1[i][2] += (player_pos_0[0] - bullets_1[i][0]) / math.pow(hy,2.5)  * gravitation
                bullets_1[i][3] += (player_pos_0[1] - bullets_1[i][1]) / math.pow(hy,2.5)  * gravitation
                
                bullets_1[i][0] += bullets_1[i][2]
                bullets_1[i][1] += bullets_1[i][3]
                i+=1
        
        # background color
        pad.fill(WHITE)

        # draw image
        player0(player_pos_0[1],player_pos_0[0])
        gun(gun_pos_0[1],gun_pos_0[0])
        for bul in bullets_0:
            bullet0(bul[1],bul[0])

        player1(player_pos_1[1],player_pos_1[0])
        gun(gun_pos_1[1],gun_pos_1[0])
        for bul in bullets_1:
            bullet1(bul[1],bul[0])

        # update
        pygame.display.update()

        # fps
        clock.tick(60)
    

def init():
    global pad, clock, aircraft0, aircraft1, gunshot, bulletshape0, bulletshape1

    pygame.init()
    pad = pygame.display.set_mode((w, h))
    pygame.display.set_caption('PVP')
    aircraft0 = pygame.image.load('images/airplane0.png')
    aircraft1 = pygame.image.load('images/airplane1.png')
    gunshot = pygame.image.load('images/gun.png')
    bulletshape0 = pygame.image.load('images/bullet0.png')
    bulletshape1 = pygame.image.load('images/bullet1.png')
    clock = pygame.time.Clock()
    
    for i in range(1):
        run()

init()