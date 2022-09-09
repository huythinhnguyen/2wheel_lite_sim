import numpy as np

HIT_DISTANCE = {1: 0.3, 2: 0.4}


def collision_check(inview, object_class):
    inview_of_klass = inview[inview[:,2]==object_class][:, :2]
    if np.sum(inview[:,0]<HIT_DISTANCE[object_class]) > 0:
        return True
    else:
        return False


def reward_function(**kwargs):
    if 'hit' in kwargs.keys():
        hit = kwargs['hit']
        if hit == 'plant': return -1
        if hit == 'pole': return +1
    else: return 0