import numpy as np
def get_distances(x0, x1, torus=False, world_size=None):
   
    delta = np.abs(x0 - x1)
    
    # print(delta.shape)
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    return dist
def get_distance_matrix(points, world_size=None, torus=False, add_to_diagonal=0):
    distance_matrix = np.vstack([get_distances(points, p, torus=torus, world_size=world_size) for p in points])
    # print(distance_matrix.shape
    return distance_matrix

if __name__ =="__main__":
    test_np=np.zeros((7,2),dtype=np.int32)
    for i in range (0,7):
        start_pos_x = np.random.randint(0, 100)
        start_pos_y = np.random.randint(0, 100)
        test_np[i,0]=start_pos_x
        
        test_np[i,1]=start_pos_y
    print(test_np)
    dist=get_distance_matrix(test_np)
    evader_dist = dist[-1,:-1]
    print(evader_dist)
    sub_list = list(np.where(evader_dist < 1.5,True,False))
    print(sub_list)
    
    # print(get_distance_matrix(test_np))
    
    