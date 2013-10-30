def get_optimal_mix_balancing(L, GW, GS, gamma=1., returnall=False, normalized=True):

        L, GW, GS = array(L,ndmin=2), array(GW,ndmin=2), array(GS,ndmin=2)  #Ensure minimum dimension to 2 to alow the weighed sum to be calculated correctly.
        weighed_sum = lambda x: sum(x,axis=0)/mean(sum(x,axis=0))

        l = weighed_sum(L)
        Gw = weighed_sum(GW)        
        Gs = weighed_sum(GS)
        
        mismatch = lambda alpha_w: gamma*(alpha_w*Gw + (1.-alpha_w)*Gs) - l
        res_load_sum = lambda alpha_w: sum(get_positive(-mismatch(alpha_w)))
        
        alpha_w_opt = fmin(res_load_sum,0.5,disp=False)
        res_load_sum_1p_interval = lambda alpha_w: res_load_sum(alpha_w)-(res_load_sum(alpha_w_opt)+.01*sum(l))
        
        alpha_w_opt_1p_interval = array([brentq(res_load_sum_1p_interval, 0, alpha_w_opt),brentq(res_load_sum_1p_interval, alpha_w_opt, 1)])
        
        if normalized:
                mismatch_opt = mismatch(alpha_w_opt)
        else:
                mismatch_opt = mismatch(alpha_w_opt)*mean(sum(L,axis=0))
        res_load_sum_opt = sum(get_positive(-mismatch_opt))
        
        if returnall:
                #Returns: alpha_w_opt, alpha_w_opt_1p_interval, res_load_sum_opt, mismatch_opt
                return alpha_w_opt, alpha_w_opt_1p_interval, res_load_sum_opt, mismatch_opt
        else:
                return alpha_w_opt
