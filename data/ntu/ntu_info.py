"""
NTU RGB+D skeleton tree (base of spine as root, start at 1)
21  2   1   17  18  19  20
            13  14  15  16
    5   6   7   8   22
                    23
    9   10  11  12  24
                    25
    3   4                
"""
PARENTS = [x - 1 for x in 
        [2, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 0, 8, 8, 12, 12]]

AVG_BONE_LENS = [0.2620958577535607, 0.19521926101937065, 0.06491050142783375, 0.10656450887708213, 
                0.11819160010358792, 0.17166537176581254, 0.10559139625675551, 0.02588212898819006, 
                0.1171968427603633, 0.1631791282751318, 0.08668784237079608, 0.023037831357693994, 
                0.06211005055063678, 0.2339934014127811, 0.30683858483618914, 0.08081869891887986, 
                0.0622302875932123, 0.23461390169899515, 0.3059475113500155, 0.08138488644070911, 
                0.0, 0.02704641233591689, 0.015233369069885367, 0.022025498895037018, 0.015830884385952617]


# Joint indices
LEFT_HAND = [8, 9, 10, 11, 23, 24]
RIGHT_HAND = [4, 5, 6, 7, 21, 22]
SPINE = [0, 1, 2, 3]
LEFT_LEG = [16, 17, 18, 19]
RIGHT_LEG = [12, 13, 14, 15]

LABEL_NAMES = [ 
'drink water',
'eat meal/snack',
'brushing teeth',
'brushing hair',
'drop',
'pickup',
'throw',
'sitting down',
'standing up (from sitting position)',
'clapping',
'reading',
'writing',
'tear up paper',
'wear jacket',
'take off jacket',
'wear a shoe',
'take off a shoe',
'wear on glasses',
'take off glasses',
'put on a hat/cap',
'take off a hat/cap',
'cheer up',
'hand waving',
'kicking something',
'reach into pocket',
'hopping (one foot jumping)',
'jump up',
'make a phone call/answer phone',
'playing with phone/tablet',
'typing on a keyboard',
'pointing to something with finger',
'taking a selfie',
'check time (from watch)',
'rub two hands together',
'nod head/bow',
'shake head',
'wipe face',
'salute',
'put the palms together',
'cross hands in front (say stop)',
'sneeze/cough',
'staggering',
'falling',
'touch head (headache)',
'touch chest (stomachache/heart pain)',
'touch back (backache)',
'touch neck (neckache)',
'nausea or vomiting condition',
'use a fan (with hand or paper)/feeling warm',
'punching/slapping other person',
'kicking other person',
'pushing other person',
'pat on back of other person',
'point finger at the other person',
'hugging other person',
'giving something to other person',
'touch other person\'s pocket',
'handshaking',
'walking towards each other',
'walking apart from each other']