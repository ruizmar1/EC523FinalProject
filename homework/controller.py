import pystk

def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=80):
    import numpy as np
    #this seems to initialize an object
    action = pystk.Action()

    # Trying to find the radian angle given by the aim point and scale that to the scale [-1, 1]
    if aim_point[0]<0:
        if aim_point[1]>0:
            action.drift = True
            action.steer = -0.9
        else:
            if aim_point[0]<-0.5:
                if aim_point[1]>-0.5:
                    action.drift = True
            else:
                action.drift = False
            
            if aim_point[0]>-0.03:
                action.steer = 0
                target_vel = 80
            elif aim_point[0]>-0.2:
                action.steer = -0.8
                target_vel = 80
            else:
                action.steer = -0.9
                target_vel = 30
    else:
        if aim_point[1]>0:
            action.drift = True
            action.steer = 0.9
        else:
            if aim_point[0]>0.5:
                if aim_point[1]>-0.5:
                    action.drift = True
            else:
                action.drift = False
            
            if aim_point[0]<0.03:
                action.steer = 0
                target_vel = 80
            elif aim_point[0]<0.2:
                action.steer = 0.8
                target_vel = 80
            else:
                action.steer = 0.9
                target_vel = 30

    # set acceleration and break to reach target velocity
    if current_vel < target_vel:
        action.brake = False
        if (current_vel/target_vel) < 0.5:
            if abs(aim_point[0])<0.1:
                action.nitro = False
        else:
            action.nitro = False
        action.acceleration = 1 - current_vel/target_vel
    elif current_vel > target_vel:
        action.brake= True

    return action

# def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
#     import numpy as np
#     #this seems to initialize an object
#     action = pystk.Action()

   
 
    

#     #compute acceleration
#     action.acceleration = 0.1
    

        

    

#     return action

    




if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
