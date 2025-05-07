# EC523FinalProject

<table>
  <tr>
    <td align="center">
      <img src="./images/IMG4_lighthouse.png" width="300px"><br>
      <b>lighthouse </b>
    </td>
    <td align="center">
      <img src="./images/IMG5_zengarden.png" width="300px"><br>
      <b>zengarden</b>
    </td>
    <td align="center">
      <img src="./images/IMG3_hacienda.png" width="300px"><br>
      <b>hacienda</b>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="./images/IMG1_snowtux.png" width="300px"><br>
      <b>snowtuxpeak</b>
    </td>
    <td align="center">
      <img src="./images/IMG2_cornfield.png" width="300px"><br>
      <b>cornfield_crossing</b>
    </td>
    <td align="center">
      <img src="./images/IMG6_scotland.png" width="300px"><br>
      <b>scotland</b>
    </td>
  </tr>
</table>

## Python Version
It is important to note that we are using a specific version of python, python 3.10. PySuperTuxKart is not compatible with newer versions of Python and the new Mac chips. As such, we had to use an emulator (in our case VS Code, here is link to download: https://code.visualstudio.com/download) to run an intel chip version of python. To specify which version of python we want to use, we added an alias in ourr .zprofiel for this specific python version which is python 3.10. For more details on this procedure, follow this tutorial: https://acrogenesis.com/running-intel-python-on-m1. 

Disclaimer for Mac M3 chips: one of our teammates could not get this workaround to work on her newer Macbook. Be warned that the above tutorial may not work for you, and in that case we have no answers (we tried for many hours though)

## Gymnasium

The reason we decided to implement Gymnasium into this project is because it will make the implementations of policy gradient methods much easier. We plan on using CleanRL pre-defiend implementations to seamlessly integrate these policy gradient methods. In this way we can minimize errors in the actual implementation and not sink too much time into debugging.

## kart_env.py

This is a custom Gymnasium wrapper written for PySuperTuxKart. We created a class called SuperTuxKartEnv with functions __init__(), reset(), step(), and render(). The init functions initalizes the environment, the reset function defines how a reset will be preformed, the step function defines how to step through the environment (i.e. take an action, pick up an award, transition to next state), and the render function displays the race using matplotlib. A lot of this code was transferred and re-used from old code in utils.py. We are just redefining the way that the environment is driven, but big picture we are not changing to much. 

![til](./images/Video-2.gif)
<center> Video of Rendered Gymnasium Enivironment </center>

 ## marissa branch
 The driver code is implemented in EC523FinalProject/homework/ddpg_continuous_action.py file. This file built off from the CleanRL implemenation of the same name: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py. We made some tweaks to make it compatible with the custom SuperTuxKart environment. This script is the driver code for this implementation. It runs 25,000 timesteps of the game environement and runs a DDPG update every 2 time steps. It also renders the game image. At the end of the run, it plots the loss over time using matplotlib. To run this file, use the following command:
```
python3.10 ddpg_continuous_action.py --track lighthouse
```
If you do not specify the track it defaults to lighthouse
- [Link to the DDPG training video](https://drive.google.com/file/d/1md7qpvwBSFCN1MlbZJqVEG3HJm2ySojz/view?usp=share_link).


 ## SOC_Bennett branch
The driver code is implemented in EC523FinalProject/project/soft_actor_critic.py file. This file built off from the CleanRL implemenation of the same name: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py. We made some tweaks to make it compatible with the custom SuperTuxKart environment. This script is the driver code for this implementation. It runs 5,000 timesteps of the game environment to build the replay buffer and trains the Q functions and actors every time step. It also renders the game image. To run this file, use the following command:
```
python3.10 soft_actor_critic.py
```
To plot the results, simply copy and paste your output into the graph.py function and run it using the following command:
```
python3.10 graph.py
```
## gabi branch
 The driver code is implemented in EC523FinalProject/homework/ppo.py file. This file built off from the CleanRL implemenation of the same name: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py. We made some tweaks to make it compatible with the custom SuperTuxKart environment. This script is the driver code for this implementation. It runs up to 50,000 timesteps of the game environement and runs the PPO update every 128 time steps. It also renders the game image during training. To run this file, use the following command:
```
python3.10 ppo.py --num_envs 1
```
If you do not specify the track it defaults to lighthouse.

To plot the results, simply copy and paste your output between the ''' ''' in the raw_data variable of the plot_output.py function, and run it using the following command:
```
python3.10 plot_output.py
```
- [Link to the PPO training video](https://drive.google.com/file/d/128smr3d9xT0DWCpyXJ1xi7RPODWwg4G7/view?usp=sharing).



