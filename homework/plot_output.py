import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === Step 1: Paste your raw log here ===
raw_data = """
Resetting the environment...
Kart crashed! Respawning...
Iteration 1
SPS: 17
Total Loss: 5332.11669921875
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 2
SPS: 19
Total Loss: 11.192662239074707
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 3
SPS: 20
Total Loss: 6.536929130554199
Kart crashed! Respawning...
Iteration 4
SPS: 20
Total Loss: 10.541666984558105
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 5
SPS: 20
Total Loss: 65.62724304199219
Kart crashed! Respawning...
Iteration 6
SPS: 20
Total Loss: 11.240859031677246
Iteration 7
SPS: 20
Total Loss: 212.25421142578125
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 8
SPS: 20
Total Loss: 10.127409934997559
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 9
SPS: 20
Total Loss: 3.9618897438049316
^CTraceback (most recent call last):
  File "/Users/gabimachado/EC523FinalProject/homework/ppo.py", line 214, in <module>
    action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
  File "/Users/gabimachado/EC523FinalProject/homework/ppo.py", line 94, in get_action_and_value
    x = self.cnn(x)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 460, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 456, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
KeyboardInterrupt

(gabriela_env) (base) gabimachado@Gabis-MacBook-Pro homework % python3.10 ppo.py --num-envs 1
[W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
Resetting the environment...
Iteration 1, SPS: 19
Total Loss: 5332.09423828125
Kart crashed! Respawning...
Iteration 2, SPS: 20
Total Loss: 0.8229408264160156
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 3, SPS: 21
Total Loss: 10.044780731201172
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 4, SPS: 21
Total Loss: 6.873019695281982
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 5, SPS: 21
Total Loss: 2.8693127632141113
Kart crashed! Respawning...
Iteration 6, SPS: 22
Total Loss: 14.362395286560059
Kart crashed! Respawning...
Iteration 7, SPS: 22
Total Loss: 7.527589321136475
Kart crashed! Respawning...
Iteration 8, SPS: 22
Total Loss: 85.12109375
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 9, SPS: 22
Total Loss: 1.401841402053833
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 10, SPS: 22
Total Loss: 0.21326139569282532
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 11, SPS: 22
Total Loss: 0.5481710433959961
Iteration 12, SPS: 22
Total Loss: 0.045594654977321625
Kart crashed! Respawning...
Iteration 13, SPS: 22
Total Loss: 1.646943211555481
Kart crashed! Respawning...
Iteration 14, SPS: 22
Total Loss: 1.127524733543396
Kart crashed! Respawning...
Iteration 15, SPS: 22
Total Loss: 2.806105136871338
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 16, SPS: 22
Total Loss: 1.6406511068344116
Kart crashed! Respawning...
Iteration 17, SPS: 22
Total Loss: 0.12098664045333862
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 18, SPS: 22
Total Loss: 0.7889717221260071
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 19, SPS: 22
Total Loss: 0.04639485850930214
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 20, SPS: 22
Total Loss: 1.8705085515975952
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 21, SPS: 22
Total Loss: 3.520035982131958
Kart crashed! Respawning...
Iteration 22, SPS: 22
Total Loss: 0.01718118041753769
Iteration 23, SPS: 22
Total Loss: 0.2658641040325165
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 24, SPS: 22
Total Loss: 4.930060863494873
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 25, SPS: 22
Total Loss: 0.2412819266319275
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 26, SPS: 22
Total Loss: 0.6849451065063477
Kart crashed! Respawning...
Iteration 27, SPS: 22
Total Loss: 0.1310797929763794
Kart crashed! Respawning...
Iteration 28, SPS: 22
Total Loss: 0.3750608265399933
Iteration 29, SPS: 22
Total Loss: -0.02453029528260231
Kart crashed! Respawning...
Iteration 30, SPS: 22
Total Loss: 4.604574203491211
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 31, SPS: 22
Total Loss: 0.3456178903579712
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 32, SPS: 22
Total Loss: 0.0043385326862335205
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 33, SPS: 22
Total Loss: 0.7981667518615723
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 34, SPS: 22
Total Loss: 0.28340280055999756
Kart crashed! Respawning...
Iteration 35, SPS: 22
Total Loss: 0.035891324281692505
Kart crashed! Respawning...
Iteration 36, SPS: 22
Total Loss: 0.08790124952793121
Kart crashed! Respawning...
Iteration 37, SPS: 22
Total Loss: 0.37071651220321655
Kart crashed! Respawning...
Iteration 38, SPS: 22
Total Loss: 0.8708981275558472
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 39, SPS: 22
Total Loss: 0.06054610013961792
Kart crashed! Respawning...
Iteration 40, SPS: 22
Total Loss: 0.7649360299110413
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 41, SPS: 22
Total Loss: -0.017348475754261017
Kart crashed! Respawning...
Iteration 42, SPS: 22
Total Loss: 0.010365167632699013
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 43, SPS: 22
Total Loss: 0.060573384165763855
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 44, SPS: 22
Total Loss: 1.7876355648040771
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 45, SPS: 22
Total Loss: 0.2426932454109192
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 46, SPS: 22
Total Loss: -0.015778059139847755
Kart crashed! Respawning...
Iteration 47, SPS: 22
Total Loss: 0.13337431848049164
Kart crashed! Respawning...
Iteration 48, SPS: 22
Total Loss: -0.04388914629817009
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 49, SPS: 22
Total Loss: 0.0910945013165474
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 50, SPS: 22
Total Loss: 0.39164215326309204
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 51, SPS: 22
Total Loss: 0.9258014559745789
Kart crashed! Respawning...
Iteration 52, SPS: 22
Total Loss: 2.578521728515625
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 53, SPS: 22
Total Loss: 0.30636969208717346
Iteration 54, SPS: 22
Total Loss: 0.02305491827428341
Kart crashed! Respawning...
Iteration 55, SPS: 22
Total Loss: 0.4789826273918152
Iteration 56, SPS: 22
Total Loss: 10.87938117980957
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 57, SPS: 22
Total Loss: 0.09639115631580353
Kart crashed! Respawning...
Iteration 58, SPS: 22
Total Loss: 0.0871242955327034
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 59, SPS: 22
Total Loss: 0.037969570606946945
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 60, SPS: 22
Total Loss: 0.4203629195690155
Kart crashed! Respawning...
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 61, SPS: 22
Total Loss: 0.19942012429237366
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 62, SPS: 22
Total Loss: 0.09273232519626617
Kart crashed! Respawning...
Iteration 63, SPS: 22
Total Loss: 0.11439836025238037
Kart crashed! Respawning...
Kart crashed! Respawning...
Iteration 64, SPS: 22
Total Loss: 0.10269465297460556
Iteration 65, SPS: 22
Total Loss: 38.332271575927734
Iteration 66, SPS: 22
Total Loss: 0.037571053951978683
Iteration 67, SPS: 22
Total Loss: 0.09214600175619125
Iteration 68, SPS: 22
Total Loss: 0.08781804144382477
Iteration 69, SPS: 22
Total Loss: 0.07272079586982727
Iteration 70, SPS: 22
Total Loss: -0.03945847228169441
Iteration 71, SPS: 22
Total Loss: 0.06394006311893463
Iteration 72, SPS: 22
Total Loss: -0.06392807513475418
Iteration 73, SPS: 22
Total Loss: 0.20248165726661682
Iteration 74, SPS: 22
Total Loss: 0.1367901861667633
Iteration 75, SPS: 22
Total Loss: 0.056245107203722
Iteration 76, SPS: 22
Total Loss: -0.00681078527122736
Iteration 77, SPS: 22
Total Loss: 0.12310431152582169
Iteration 78, SPS: 22
Total Loss: 0.02258043922483921
Iteration 79, SPS: 22
Total Loss: -0.025225860998034477
Iteration 80, SPS: 22
Total Loss: 0.07034074515104294
Iteration 81, SPS: 22
Total Loss: 0.02815541811287403
Iteration 82, SPS: 22
Total Loss: 0.11384297162294388
Iteration 83, SPS: 22
Total Loss: 0.06799276173114777
Iteration 84, SPS: 22
Total Loss: 0.15359468758106232
Iteration 85, SPS: 22
Total Loss: -0.07641040533781052
Iteration 86, SPS: 22
Total Loss: -0.0763917863368988
Iteration 87, SPS: 22
Total Loss: 0.1223485991358757
Iteration 88, SPS: 22
Total Loss: 0.14783157408237457
Iteration 89, SPS: 22
Total Loss: 0.12005627155303955
Iteration 90, SPS: 22
Total Loss: 0.06424006074666977
Iteration 91, SPS: 22
Total Loss: 0.06340032070875168
Iteration 92, SPS: 22
Total Loss: 0.16037604212760925
Iteration 93, SPS: 22
Total Loss: 0.11705741286277771
Iteration 94, SPS: 22
Total Loss: 0.11334624141454697
Iteration 95, SPS: 22
Total Loss: -0.04558884724974632
Iteration 96, SPS: 22
Total Loss: 0.18908579647541046
Iteration 97, SPS: 22
Total Loss: 0.052133794873952866
Iteration 98, SPS: 22
Total Loss: 0.11011429876089096
Iteration 99, SPS: 22
Total Loss: -0.0617450512945652
Iteration 100, SPS: 22
Total Loss: 0.05693450942635536
Iteration 101, SPS: 22
Total Loss: -0.06927155703306198
Iteration 102, SPS: 22
Total Loss: -0.009009849280118942
Iteration 103, SPS: 22
Total Loss: 0.0528767853975296
Iteration 104, SPS: 22
Total Loss: -0.030900223180651665
Iteration 105, SPS: 22
Total Loss: 0.027961017563939095
Iteration 106, SPS: 22
Total Loss: 0.08378780633211136
Iteration 107, SPS: 22
Total Loss: 0.07434052228927612
Iteration 108, SPS: 22
Total Loss: -0.03438551351428032
Iteration 109, SPS: 22
Total Loss: 0.1106758713722229
Iteration 110, SPS: 22
Total Loss: 0.0039429208263754845
Iteration 111, SPS: 22
Total Loss: -0.07356388121843338
Iteration 112, SPS: 22
Total Loss: -0.00226355018094182
Iteration 113, SPS: 22
Total Loss: -0.024132343009114265
Iteration 114, SPS: 22
Total Loss: -0.04536556825041771
Iteration 115, SPS: 22
Total Loss: 0.0010661300038918853
Iteration 116, SPS: 22
Total Loss: -0.0026976806111633778
Iteration 117, SPS: 22
Total Loss: 0.05218309164047241
Iteration 118, SPS: 22
Total Loss: -0.007036304567009211
Iteration 119, SPS: 22
Total Loss: -0.010046190582215786
Iteration 120, SPS: 22
Total Loss: 0.11021412909030914
Iteration 121, SPS: 22
Total Loss: 0.06663420051336288
Iteration 122, SPS: 22
Total Loss: -0.07349864393472672
Iteration 123, SPS: 22
Total Loss: -0.02503145858645439
Iteration 124, SPS: 22
Total Loss: -0.038876552134752274
Iteration 125, SPS: 22
Total Loss: 0.09395627677440643
Iteration 126, SPS: 22
Total Loss: 0.053957801312208176
Iteration 127, SPS: 22
Total Loss: -0.04873098433017731
Iteration 128, SPS: 22
Total Loss: 0.049221109598875046
Iteration 129, SPS: 22
Total Loss: 0.05510533228516579
Iteration 130, SPS: 22
Total Loss: 0.10464693605899811
Iteration 131, SPS: 22
Total Loss: 0.16918988525867462
Iteration 132, SPS: 22
Total Loss: -0.013910247012972832
Iteration 133, SPS: 22
Total Loss: 0.12622100114822388
Iteration 134, SPS: 22
Total Loss: -0.01809437945485115
Iteration 135, SPS: 22
Total Loss: 0.11860208213329315
Iteration 136, SPS: 22
Total Loss: -0.06996508687734604
Iteration 137, SPS: 22
Total Loss: 0.022244980558753014
Iteration 138, SPS: 22
Total Loss: -0.04371662437915802
Iteration 139, SPS: 22
Total Loss: -0.06515602022409439
Iteration 140, SPS: 22
Total Loss: 0.11035002022981644
Iteration 141, SPS: 22
Total Loss: -0.006065583787858486
Iteration 142, SPS: 22
Total Loss: 0.01619516871869564
Iteration 143, SPS: 22
Total Loss: 0.0048692310228943825
Iteration 144, SPS: 22
Total Loss: 0.014705189503729343
Iteration 145, SPS: 22
Total Loss: -0.010289008729159832
Iteration 146, SPS: 22
Total Loss: 0.032333169132471085
Iteration 147, SPS: 22
Total Loss: 0.03423488512635231
Iteration 148, SPS: 22
Total Loss: 0.028045665472745895
Iteration 149, SPS: 22
Total Loss: -0.043014585971832275
Iteration 150, SPS: 22
Total Loss: -0.028388628736138344
Iteration 151, SPS: 22
Total Loss: 0.030689287930727005
Iteration 152, SPS: 22
Total Loss: -0.010487934574484825
Iteration 153, SPS: 22
Total Loss: -0.03671351075172424
Iteration 154, SPS: 22
Total Loss: -0.025355730205774307
Iteration 155, SPS: 22
Total Loss: -0.0151173435151577
Iteration 156, SPS: 22
Total Loss: 0.04061704874038696
Iteration 157, SPS: 22
Total Loss: 0.01126893237233162
Iteration 158, SPS: 22
Total Loss: -0.024197012186050415
Iteration 159, SPS: 22
Total Loss: 0.08003193140029907
Iteration 160, SPS: 22
Total Loss: 0.0006738235824741423
Iteration 161, SPS: 22
Total Loss: 0.07642975449562073
Iteration 162, SPS: 22
Total Loss: 0.0073882476426661015
Iteration 163, SPS: 22
Total Loss: -0.044379666447639465
Iteration 164, SPS: 22
Total Loss: 0.004024735186249018
Iteration 165, SPS: 22
Total Loss: -0.08036177605390549
Iteration 166, SPS: 22
Total Loss: -0.03794838860630989
Iteration 167, SPS: 22
Total Loss: 0.02286357246339321
Iteration 168, SPS: 22
Total Loss: 0.06491909176111221
Iteration 169, SPS: 22
Total Loss: -0.009214768186211586
Iteration 170, SPS: 22
Total Loss: -0.045846305787563324
Iteration 171, SPS: 22
Total Loss: -0.0131189851090312
Iteration 172, SPS: 22
Total Loss: -0.053190186619758606
Iteration 173, SPS: 22
Total Loss: 0.028062129393219948
Iteration 174, SPS: 22
Total Loss: -0.019251668825745583
Iteration 175, SPS: 22
Total Loss: -0.03526265546679497
Iteration 176, SPS: 22
Total Loss: -0.016945375129580498
Iteration 177, SPS: 22
Total Loss: -0.10675746947526932
Iteration 178, SPS: 22
Total Loss: 0.03428472951054573
Iteration 179, SPS: 22
Total Loss: -0.04271957650780678
Iteration 180, SPS: 22
Total Loss: 0.05983849987387657
Iteration 181, SPS: 22
Total Loss: 0.016238512471318245
Iteration 182, SPS: 22
Total Loss: -0.11774519085884094
Iteration 183, SPS: 22
Total Loss: 0.12681454420089722
Iteration 184, SPS: 22
Total Loss: -0.050627339631319046
Iteration 185, SPS: 22
Total Loss: -0.10341013967990875
Iteration 186, SPS: 22
Total Loss: -0.06087874993681908
Iteration 187, SPS: 22
Total Loss: 0.05441465228796005
Iteration 188, SPS: 22
Total Loss: -0.013846074230968952
Iteration 189, SPS: 22
Total Loss: 0.10246448218822479
Iteration 190, SPS: 22
Total Loss: -0.09119248390197754
Iteration 191, SPS: 22
Total Loss: -0.028324302285909653
Iteration 192, SPS: 22
Total Loss: -0.07261475920677185
Iteration 193, SPS: 22
Total Loss: -0.027203310281038284
Iteration 194, SPS: 22
Total Loss: -0.03565196320414543
Iteration 195, SPS: 22
Total Loss: -0.07671938091516495
Iteration 196, SPS: 22
Total Loss: 0.004380419384688139
Iteration 197, SPS: 22
Total Loss: 0.019508380442857742
Iteration 198, SPS: 22
Total Loss: -0.019088974222540855
Iteration 199, SPS: 22
Total Loss: -0.008265837095677853
Iteration 200, SPS: 22
Total Loss: -0.13730469346046448
Iteration 201, SPS: 22
Total Loss: -0.0956205353140831
Iteration 202, SPS: 22
Total Loss: -0.0393742136657238
Iteration 203, SPS: 22
Total Loss: -0.05695370212197304
Iteration 204, SPS: 22
Total Loss: -0.06349585205316544
Iteration 205, SPS: 22
Total Loss: 0.026168016716837883
Iteration 206, SPS: 22
Total Loss: 0.039454683661460876
Iteration 207, SPS: 22
Total Loss: -0.10441910475492477
Iteration 208, SPS: 22
Total Loss: -0.07050371170043945
Iteration 209, SPS: 22
Total Loss: -0.03789658099412918
Iteration 210, SPS: 22
Total Loss: -0.037939660251140594
Iteration 211, SPS: 22
Total Loss: -0.04578480124473572
Iteration 212, SPS: 22
Total Loss: 0.003607043530791998
Iteration 213, SPS: 22
Total Loss: 0.033594075590372086
Iteration 214, SPS: 22
Total Loss: 0.06101199984550476
Iteration 215, SPS: 22
Total Loss: -0.10258006304502487
Iteration 216, SPS: 22
Total Loss: -0.0429670624434948
Iteration 217, SPS: 22
Total Loss: 0.011760352179408073
Iteration 218, SPS: 22
Total Loss: 0.08496147394180298
Iteration 219, SPS: 22
Total Loss: 0.02513767033815384
Iteration 220, SPS: 22
Total Loss: 0.0074555762112140656
Iteration 221, SPS: 22
Total Loss: 0.0463024266064167
Iteration 222, SPS: 22
Total Loss: -0.020497525110840797
Iteration 223, SPS: 22
Total Loss: -0.05145599693059921
Iteration 224, SPS: 22
Total Loss: 0.014752653427422047
Iteration 225, SPS: 22
Total Loss: 0.02836652286350727
Iteration 226, SPS: 22
Total Loss: 0.016342151910066605
Iteration 227, SPS: 22
Total Loss: 0.18669584393501282
Iteration 228, SPS: 22
Total Loss: 0.07481616735458374
Iteration 229, SPS: 22
Total Loss: -0.032157812267541885
Iteration 230, SPS: 22
Total Loss: -0.013383334502577782
Iteration 231, SPS: 22
Total Loss: 0.03176559507846832
Iteration 232, SPS: 22
Total Loss: -0.06494168192148209
Iteration 233, SPS: 22
Total Loss: -0.02936902828514576
Iteration 234, SPS: 22
Total Loss: 0.007980013266205788
Iteration 235, SPS: 22
Total Loss: -0.09292218089103699
Iteration 236, SPS: 22
Total Loss: -0.018569493666291237
Iteration 237, SPS: 22
Total Loss: -0.05086127296090126
Iteration 238, SPS: 22
Total Loss: -0.03610430285334587
Iteration 239, SPS: 22
Total Loss: 0.0978841483592987
Iteration 240, SPS: 22
Total Loss: 0.0064995987340807915
Iteration 241, SPS: 22
Total Loss: 0.1255923956632614
Iteration 242, SPS: 22
Total Loss: -0.040971897542476654
Iteration 243, SPS: 22
Total Loss: -0.02269839309155941
Iteration 244, SPS: 22
Total Loss: -0.0013363875914365053
Iteration 245, SPS: 22
Total Loss: 0.06926580518484116
Iteration 246, SPS: 22
Total Loss: -0.002219012938439846
Iteration 247, SPS: 22
Total Loss: 0.020913681015372276
Iteration 248, SPS: 22
Total Loss: 0.0012762828264385462
Iteration 249, SPS: 22
Total Loss: -0.00214129569940269
Iteration 250, SPS: 22
Total Loss: -0.02360086515545845
Iteration 251, SPS: 22
Total Loss: -0.08634170889854431
Iteration 252, SPS: 22
Total Loss: -0.07208579033613205
Iteration 253, SPS: 22
Total Loss: -0.09465938806533813
Iteration 254, SPS: 22
Total Loss: -0.06684249639511108
Iteration 255, SPS: 22
Total Loss: -0.05990651994943619
Iteration 256, SPS: 22
Total Loss: -0.04676136001944542
Iteration 257, SPS: 22
Total Loss: 0.04815765842795372
Iteration 258, SPS: 22
Total Loss: -0.06642619520425797
Iteration 259, SPS: 22
Total Loss: -0.02452508918941021
Iteration 260, SPS: 22
Total Loss: -0.08781953901052475
Iteration 261, SPS: 22
Total Loss: 0.033680934458971024
Iteration 262, SPS: 22
Total Loss: -0.08196889609098434
Iteration 263, SPS: 22
Total Loss: -0.08614856004714966
Iteration 264, SPS: 22
Total Loss: 0.03039385937154293
Iteration 265, SPS: 22
Total Loss: -0.16364967823028564
Iteration 266, SPS: 22
Total Loss: -0.07936450093984604
Iteration 267, SPS: 22
Total Loss: -0.05130412429571152
Iteration 268, SPS: 22
Total Loss: 0.007759119383990765
Iteration 269, SPS: 22
Total Loss: 0.04353959485888481
Iteration 270, SPS: 22
Total Loss: 0.0023004126269370317
Iteration 271, SPS: 22
Total Loss: 0.011383946985006332
Iteration 272, SPS: 22
Total Loss: -0.036429714411497116
Iteration 273, SPS: 22
Total Loss: -0.025530891492962837
Iteration 274, SPS: 22
Total Loss: -0.009796731173992157
Iteration 275, SPS: 22
Total Loss: -0.034905146807432175
Iteration 276, SPS: 22
Total Loss: 0.051785361021757126
Iteration 277, SPS: 22
Total Loss: -0.007062029559165239
Iteration 278, SPS: 22
Total Loss: -0.035834215581417084
Iteration 279, SPS: 22
Total Loss: -0.09875304996967316
Iteration 280, SPS: 22
Total Loss: -0.057529330253601074
Iteration 281, SPS: 22
Total Loss: -0.10109735280275345
Iteration 282, SPS: 22
Total Loss: -0.038210947066545486
Iteration 283, SPS: 22
Total Loss: 0.034925833344459534
Iteration 284, SPS: 22
Total Loss: -0.09167370945215225
Iteration 285, SPS: 22
Total Loss: -0.101609967648983
Iteration 286, SPS: 22
Total Loss: -0.0006230504368431866
Iteration 287, SPS: 22
Total Loss: -0.029508884996175766
Iteration 288, SPS: 22
Total Loss: 0.048021841794252396
Iteration 289, SPS: 22
Total Loss: -0.002272749086841941
Iteration 290, SPS: 22
Total Loss: -0.12252216041088104
Iteration 291, SPS: 22
Total Loss: -0.02375674806535244
Iteration 292, SPS: 22
Total Loss: -0.06817547231912613
Iteration 293, SPS: 22
Total Loss: -0.016781575977802277
Iteration 294, SPS: 22
Total Loss: -0.08697724342346191
Iteration 295, SPS: 22
Total Loss: -0.04697849601507187
Iteration 296, SPS: 22
Total Loss: 0.03198128193616867
Iteration 297, SPS: 22
Total Loss: -0.08858273178339005
Iteration 298, SPS: 22
Total Loss: 0.027230434119701385
Iteration 299, SPS: 22
Total Loss: 0.04147125780582428
Iteration 300, SPS: 22
Total Loss: -0.02588239684700966
Iteration 301, SPS: 22
Total Loss: 0.0033663040958344936
Iteration 302, SPS: 22
Total Loss: -0.04842667281627655
Iteration 303, SPS: 22
Total Loss: -0.08924344927072525
Iteration 304, SPS: 22
Total Loss: -0.03103642351925373
Iteration 305, SPS: 22
Total Loss: 0.00330963428132236
Iteration 306, SPS: 22
Total Loss: -0.10450423508882523
Iteration 307, SPS: 22
Total Loss: -0.1336766481399536
Iteration 308, SPS: 22
Total Loss: 0.013046830892562866
Iteration 309, SPS: 22
Total Loss: -0.06652576476335526
Iteration 310, SPS: 22
Total Loss: -0.07754253596067429
Iteration 311, SPS: 22
Total Loss: 0.07346383482217789
Iteration 312, SPS: 22
Total Loss: -0.050548333674669266
Iteration 313, SPS: 22
Total Loss: 0.04697543755173683
Iteration 314, SPS: 22
Total Loss: -0.05570124834775925
Iteration 315, SPS: 22
Total Loss: -0.06967750191688538
Iteration 316, SPS: 22
Total Loss: -0.06223033368587494
Iteration 317, SPS: 22
Total Loss: -0.13561417162418365
Iteration 318, SPS: 22
Total Loss: -0.017610130831599236
Iteration 319, SPS: 22
Total Loss: 0.06589304655790329
Iteration 320, SPS: 22
Total Loss: 0.01670330949127674
Iteration 321, SPS: 22
Total Loss: 0.03543087840080261
Iteration 322, SPS: 22
Total Loss: 0.008266519755125046
Iteration 323, SPS: 22
Total Loss: -0.08823731541633606
Iteration 324, SPS: 22
Total Loss: 0.04657285287976265
Iteration 325, SPS: 22
Total Loss: 0.002232912927865982
Iteration 326, SPS: 22
Total Loss: 0.012921376153826714
Iteration 327, SPS: 22
Total Loss: -0.1482442021369934
Iteration 328, SPS: 22
Total Loss: 0.0412735678255558
Iteration 329, SPS: 22
Total Loss: -0.02005578950047493
Iteration 330, SPS: 22
Total Loss: 0.015358834527432919
Iteration 331, SPS: 22
Total Loss: -0.0109181459993124
Iteration 332, SPS: 22
Total Loss: 0.0022854190319776535
Iteration 333, SPS: 22
Total Loss: -0.021134216338396072
Iteration 334, SPS: 22
Total Loss: -0.023192554712295532
Iteration 335, SPS: 22
Total Loss: -0.07294445484876633
Iteration 336, SPS: 22
Total Loss: 0.03236960247159004
Iteration 337, SPS: 22
Total Loss: -0.036740440875291824
Iteration 338, SPS: 22
Total Loss: -0.01707538776099682
Iteration 339, SPS: 22
Total Loss: -0.014230445958673954
Iteration 340, SPS: 22
Total Loss: 0.028307653963565826
Iteration 341, SPS: 22
Total Loss: -0.010508310981094837
Iteration 342, SPS: 22
Total Loss: -0.08825356513261795
Iteration 343, SPS: 22
Total Loss: -0.10723861306905746
Iteration 344, SPS: 22
Total Loss: -0.06506316363811493
Iteration 345, SPS: 22
Total Loss: -0.02470454014837742
Iteration 346, SPS: 22
Total Loss: 0.011705194599926472
Iteration 347, SPS: 22
Total Loss: -0.043056316673755646
Iteration 348, SPS: 22
Total Loss: -0.00028995811589993536
Iteration 349, SPS: 22
Total Loss: -0.0851210430264473
Iteration 350, SPS: 22
Total Loss: 0.07512273639440536
Iteration 351, SPS: 22
Total Loss: -0.0030799172818660736
Iteration 352, SPS: 22
Total Loss: -0.0377025231719017
Iteration 353, SPS: 22
Total Loss: 0.0013730957871302962
Iteration 354, SPS: 22
Total Loss: 0.02449832484126091
Iteration 355, SPS: 22
Total Loss: 0.015057058073580265
Iteration 356, SPS: 22
Total Loss: -0.08597268909215927
Iteration 357, SPS: 22
Total Loss: -0.0514957420527935
Iteration 358, SPS: 22
Total Loss: -0.028677964583039284
Iteration 359, SPS: 22
Total Loss: -0.010771490633487701
Iteration 360, SPS: 22
Total Loss: 0.024440212175250053
Iteration 361, SPS: 22
Total Loss: -0.018821898847818375
Iteration 362, SPS: 22
Total Loss: -0.10021679848432541
Iteration 363, SPS: 22
Total Loss: -0.02337855100631714
Iteration 364, SPS: 22
Total Loss: -0.04595149680972099
Iteration 365, SPS: 22
Total Loss: -0.03360271453857422
Iteration 366, SPS: 22
Total Loss: -0.06748366355895996
Iteration 367, SPS: 22
Total Loss: -0.04378357157111168
Iteration 368, SPS: 22
Total Loss: -0.1291341930627823
çIteration 369, SPS: 22
Total Loss: -0.11603734642267227
"""

# === Step 2: Extract Total Loss values ===
total_losses = [float(match) for match in re.findall(r"Total Loss: ([\-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", raw_data)]

# === Step 3: Plot raw total loss ===
plt.figure(figsize=(10, 6))
plt.plot(total_losses, label="Total Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Losses During PPO Training")
plt.legend()
plt.grid()
#plt.ylim(-1, 1)
plt.show()

# === Step 4: Apply smoothing and plot again ===
smoothed_loss = gaussian_filter1d(total_losses, sigma=2)

plt.figure(figsize=(10, 6))
plt.plot(smoothed_loss, label="Smoothed Total Loss", color="red")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Smoothed Losses During PPO Training")
plt.legend()
plt.grid()
#plt.ylim(-1, 1)
plt.show()
