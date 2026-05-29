import torch

def symmetrical_function(obs: torch.Tensor, actions: torch.Tensor) -> dict:
    """if want use symmetrical loss, you must define the symmetrical function 
    to calculate symmetrical_obs and symmetrical_actions
    @input param:
    obs: observation batch
    actions: actions batch
    
    @output param -> dict
    symmetrical_obs: symmetrical_obs
    symmetrical_actions: symmetrical_actions"""

    obs_cat = None
    symmetrical_actions = None

    if obs is not None:
        #Note: we need to use the obs grad, so you must use obs.clone, not obs.
        symmetrical_obs = obs.clone()
        #leg joint position obs changed
        symmetrical_obs[:, 9:15], symmetrical_obs[:, 15:21] = symmetrical_obs[:, 15:21], symmetrical_obs[:, 9:15]
        #leg joint velocity obs changed
        symmetrical_obs[:, 21:27], symmetrical_obs[:, 27:33] = symmetrical_obs[:, 27:33], symmetrical_obs[:, 21:27]
        #last actions obs changed(save)
        # symmetrical_obs[:, 33: 39], symmetrical_obs[:, 39:45] = symmetrical_obs[:, 39:45], symmetrical_obs[:, 33:39]
        #obs cat:
        obs_cat = torch.cat((obs, symmetrical_obs), dim=0)

    if actions is not None:
        #it is no need to calculate the actions grad, so we use detach() here.
        symmetrical_actions = actions.detach()
        #leg's actions changed
        symmetrical_actions[:, :6] , symmetrical_actions[:, 6:12] = symmetrical_actions[:, 6:12], symmetrical_actions[:, :6]

    return (obs_cat, symmetrical_actions)

def symmetrical_function2(obs: torch.Tensor, actions: torch.Tensor) -> dict:
    """if want use symmetrical loss, you must define the symmetrical function 
    to calculate symmetrical_obs and symmetrical_actions
    @input param:
    obs: observation batch
    actions: actions batch
    
    @output param -> dict
    symmetrical_obs: symmetrical_obs
    symmetrical_actions: symmetrical_actions"""

    obs_cat = None
    symmetrical_actions = None

    if obs is not None:
        #Note: we need to use the obs grad, so you must use obs.clone, not obs.
        symmetrical_obs = obs.clone()
        #leg joint position obs changed
        symmetrical_obs[:, 12], symmetrical_obs[:, 16] = -symmetrical_obs[:, 16], -symmetrical_obs[:, 12]
        symmetrical_obs[:, 13:15], symmetrical_obs[:, 17:19] = symmetrical_obs[:, 17:19], symmetrical_obs[:, 13:15]
        #leg joint velocity obs changed
        symmetrical_obs[:, 19:22], symmetrical_obs[:, 23:26] = symmetrical_obs[:, 23:26], symmetrical_obs[:, 19:22]
        #last actions obs changed(save)
        #obs cat:
        obs_cat = torch.cat((obs, symmetrical_obs), dim=0)

    if actions is not None:
        #it is no need to calculate the actions grad, so we use detach() here.
        symmetrical_actions = actions.detach()
        #leg's actions changed
        symmetrical_actions[:, 0] , symmetrical_actions[:, 4] = -symmetrical_actions[:, 4], -symmetrical_actions[:, 0]
        symmetrical_actions[:, 1:3] , symmetrical_actions[:, 5:7] = symmetrical_actions[:, 5:7], symmetrical_actions[:, 1:3]

    return (obs_cat, symmetrical_actions)


