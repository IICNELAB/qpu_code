"""
FPHA skeleton tree (wrist as root, start at 0)
0
    1   6   7   8
    2   9   10  11
    3   12  13  14
    4   15  16  17
    5   18  19  20
"""
PARENTS = [-1, 0, 0, 0, 0, 0, 1, 6, 7, 2, 9, 10, 3, 12, 13, 4, 15, 16, 5, 18, 19]

AVG_BONE_LENS = {0: [10.142819937239926, 61.73668767616669, 59.57166393036227, 57.02207071846138, 54.75446126107755], 
                 1: [47.84382424587107], 
                 2: [38.17755136987607], 
                 3: [39.78069497606911], 
                 4: [35.18236460692162], 
                 5: [28.435481921321824], 
                 6: [32.16864540116016], 
                 7: [21.953679510919574], 
                 8: [], 
                 9: [20.211191189483085], 
                 10: [12.968750212685839], 
                 11: [], 
                 12: [19.536761526020157], 
                 13: [12.247010910298874], 
                 14: [], 
                 15: [17.4477001039134], 
                 16: [11.707253016457981], 
                 17: [], 
                 18: [15.106989304194864], 
                 19: [9.987799490039151], 
                 20: []}

# Joint indices
T = [1, 6, 7, 8]
I = [2, 9, 10, 11]
M = [3, 12, 13, 14]
R = [4, 15, 16, 17]
P = [5, 18, 19, 20]

LABEL_NAMES = ['open_juice_bottle',
'close_juice_bottle',
'pour_juice_bottle',
'open_peanut_butter',
'close_peanut_butter',
'prick',
'sprinkle',
'scoop_spoon',
'put_sugar',
'stir',
'open_milk',
'close_milk',
'pour_milk',
'drink_mug',
'put_tea_bag',
'put_salt',
'open_liquid_soap',
'close_liquid_soap',
'pour_liquid_soap',
'wash_sponge',
'flip_sponge',
'scratch_sponge',
'squeeze_sponge',
'open_soda_can',
'use_flash',
'write',
'tear_paper',
'squeeze_paper',
'open_letter',
'take_letter_from_enveloppe',
'read_letter',
'flip_pages',
'use_calculator',
'light_candle',
'charge_cell_phone',
'unfold_glasses',
'clean_glasses',
'open_wallet',
'give_coin',
'receive_coin',
'give_card',
'pour_wine',
'toast_wine',
'handshake',
'high_five']