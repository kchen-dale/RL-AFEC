import numpy as np
import copy

frame_size_min=[55435.000000, 18715.000000, 18038.000000, 17693.000000, 17916.000000, 
17864.000000, 17835.000000, 18000.000000, 18098.000000, 18431.000000, 18264.000000, 
18499.000000, 18097.000000, 18040.000000, 18374.000000, 18549.000000, 18314.000000, 
18581.000000, 19016.000000, 18847.000000, 18110.000000, 18141.000000, 18327.000000,
18077.000000, 18067.000000, 18425.000000, 18693.000000, 18880.000000, 18304.000000, 17986.000000]
frame_size_max=[115038.000000, 78933.000000, 78295.000000, 77354.000000, 76308.000000, 76546.000000,
75867.000000, 76520.000000, 75977.000000, 75992.000000, 76137.000000, 75772.000000, 75788.000000,
75451.000000, 75992.000000, 76223.000000, 76097.000000, 76377.000000, 75631.000000, 76094.000000,
76223.000000, 76533.000000, 76442.000000, 76483.000000, 76527.000000, 76016.000000, 76310.000000,
76174.000000, 76363.000000, 75966.000000]
def decode(loss_pattern_before, percentage, critical_num):
    # Critical frames correspond to percentage[0] - percentage[K-1], non-critical frames share the same percentage[K]
    # K is the number of critical frames
    # loss pattern description: 
    # 0-29d: loss_rate (# lost pkts/ total pkts) per frame, 
    # 30-59d: frame size per frame normalized, 
    # 60-89: motion vector
    recover_pattern = copy.copy(loss_pattern_before)
    # for each frame, compare the number of packets lost and the number of redundant packets added
    # If the number of redundant packets is larger than the number of lost packets, then all lost packets in the frame can be recovered.
    bw_waste = 0
    for i in range(30):
        frame_size = frame_size_min[i] + loss_pattern_before[i+30]*(frame_size_max[i] - frame_size_min[i])
        if i < critical_num:
            num_pkt_add = int( (percentage[i] * frame_size) / 1500 ) + 1# Size of each packet is 1500 bytes
        else:
            num_pkt_add = int( (percentage[critical_num] * frame_size) / 1500 ) + 1# Size of each packet is 1500 bytes

        num_pkt_lost = int( (loss_pattern_before[i] * frame_size) / 1500 )
        if num_pkt_add >= num_pkt_lost:
            recover_pattern[i] = 0
            bw_waste = bw_waste + (num_pkt_add - num_pkt_lost) * 1500
        else:
            bw_waste = bw_waste + num_pkt_add * 1500
        
    return recover_pattern, bw_waste
