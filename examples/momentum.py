import numpy as np


def myTradingSystem(DATE,CLOSE,settings):

    p = np.zeros(np.shape(CLOSE[-1,:]))

    momentum = (CLOSE[-1,:] - CLOSE[-100,:]) / CLOSE[-1,:]
    p[momentum > 0] = 1
    p[momentum < 0] = -0.2

    positions =  p/np.sum(p)

    return positions, settings


def mySettings():
    settings={}
    settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
     'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
     'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
     'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
     'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W',
     'F_XX', 'F_YM']

    settings['slippage'] = 0.05
    settings['budget'] = 10**6
    settings['lookback'] = 504
    return settings
