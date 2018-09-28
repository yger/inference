from brian2 import * 
import pandas as pd
import numpy as np

def if_network(params,case='exponential',adapt=False,syn='current',synname=None):

    np.random.seed(100)
    defaultclock.dt = 0.05*ms
    set_device('cpp_standalone')
    devices.device.seed(seed=111)
    
    device.reinit()
    device.activate()

    ######################
    ##### PARAMETERS #####
    ######################
    Nneurons_tot = params['N'] # number of neurons in total;
    sim_t = params['t']*second #simulation time
    N_e = int(0.8*Nneurons_tot) #number of excitatory neurons
    N_i = Nneurons_tot - N_e #number of inhibitory neurons
    
    EL = params['EL'] * mV
    Vr = params['V_r'] *mV
    VT = params['V_s'] * mV
    T_ref = params['T_ref'] * ms
    C = params['C'] * pF
    gL = params['gL'] * nS
    
    # type
    if case=='exponential':
        DeltaT = params['Delta_T'] * mV
        Vcut = VT + 5 * DeltaT 
        
        # external stimuli
        rate = params['rate'] * Hz
        
        ext_weight = params['ext_weight']
    
    else:
        taum = params['tau_m']*ms
    
    # adaptation
    if adapt==True:
    
        tauw = params['tauw'] *ms
        a = params['a'] * nS
        b = params['b'] *nA
        DeltaW = params['Delta_w'] * mV/ms
     
    
    # Synapses
    pe = params['pe'] #excitatory connection probability
    pi = params['pi'] #inhibitory connection probability
    d = params['d'] #synaptic delay
    
    we_init = params['we_init']
    wi_init = params['wi_init']
    taue = params['taue'] * ms
    taui = params['taui'] * ms
    
    if syn=='current':
        
        if case=='exponential':
            taum = C / gL
            if adapt==True:
            
                eqs = '''
                    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + gL*(ge + gi) - w)/C : volt
                    dw/dt = (a*(vm - EL) - w)/tauw : amp 
                    dge/dt = -ge/taue : volt
                    dgi/dt = -gi/taui : volt
                    '''
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='euler',
                        threshold='vm>Vcut',
                        reset="vm=Vr; w+=b",
                        refractory=T_ref
                       )
            else:
            
                eqs = '''
                    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + gL*(ge + gi))/C : volt
                    dge/dt = -ge/taue : volt
                    dgi/dt = -gi/taui : volt
                    '''
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='euler',
                        threshold='vm>Vcut',
                        reset="vm=Vr",
                        refractory=T_ref
                       )
            
            # intial values
            G.vm = EL + (VT - EL)*np.random.rand(Nneurons_tot)
            G.ge = 0.0
            G.gi = 0.0
            # external noise
            P = PoissonInput(G, 'ge', N=1000, rate=rate, weight=ext_weight*mV)
        
        else:
            if adapt==True:
                eqs = '''
                    dvm/dt  = (ge+gi-(vm-EL))/taum - w: volt (unless refractory)
                    dw/dt = - w/tauw : volt/second
                    dge/dt = -ge/taue : volt
                    dgi/dt = -gi/taui : volt    
                    '''
    
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='exact',
                            threshold='vm>VT',
                            reset='vm=Vr; w+=DeltaW',
                            refractory=T_ref)
                G.w = 0.0
                
            else:
                eqs = '''
                    dvm/dt  = (ge+gi-(vm-EL))/taum : volt (unless refractory)
                    dge/dt = -ge/taue : volt
                    dgi/dt = -gi/taui : volt    
                    '''
    
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='exact',
                            threshold='vm>VT',
                            reset='vm=Vr',
                            refractory=T_ref)
    
            # intial values
            G.vm = 'Vr + rand() * (VT - Vr)'
            G.ge = 0.0
            G.gi = 0.0
            
        # synapses
        Se = Synapses(G[:N_e], G, 'we: volt', on_pre='ge += we')
        Si = Synapses(G[N_e:], G, 'wi: volt', on_pre='gi += wi')
        Se.connect('i!=j', p=pe)
        Si.connect('i+N_e!=j', p=pi)
        if synname is None:
            
            Se.we = "we_init*mV*rand()" # random excitatory synaptic weight (voltage)
            Si.wi = "wi_init*mV*rand()" # random inhibitory synaptic weight
        else:
            # load excisting synapses
            synapses = pd.read_pickle(synname)
            
            # connect
            #Se = Synapses(G[synapses['ex_i'][0]],G[synapses['ex_j'][0]],'we:volt',on_pre='ge +=we')
            #Si = Synapses(G[synapses['inh_i'][0]],G[synapses['inh_j'][0]],'wi:volt',on_pre='gi +=wi')
            Se.we = synapses['ex_we'][0]*mV
            Si.wi = synapses['inh_wi'][0]*mV
            
    
    else:
        # Reversal potentials
        Ee = params['Ee'] * mV #reversal potential of excitatory synapses
        Ei = params['Ei'] * mV
        
        if case=='exponential':
            taum = C / gL
            if adaptation==True:
                eqs = '''
                        dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) - ge*(vm - Ee) -gi*(vm - Ei) - w)/C : volt
                        dw/dt = (a*(vm - EL) - w)/tauw : amp 
                        dge/dt = -ge/taue : siemens
                        dgi/dt = -gi/taui : siemens
                        '''
                    # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='euler',
                        threshold='vm>Vcut',
                        reset="vm=Vr; w+=b",
                        refractory=T_ref
                       )
            else:
                eqs = '''
                        dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) - ge*(vm - Ee) -gi*(vm - Ei))/C : volt
                        dge/dt = -ge/taue : siemens
                        dgi/dt = -gi/taui : siemens
                        '''
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='euler',
                        threshold='vm>Vcut',
                        reset="vm=Vr",
                        refractory=T_ref
                       )
                
            # initial values    
            G.vm = EL + (VT - EL)*np.random.rand(Nneurons_tot)
            # external noise
            P = PoissonInput(G, 'ge', N=1000, rate=rate, weight=ext_weight*nS) 
            
        else:
            taum = C / gL
            if adapt==True:
                eqs = '''
                    dvm/dt  = (- gL*(vm-EL) -ge*(vm - Ee) - gi*(vm - Ei) - w)/C: volt (unless refractory)
                    dw/dt = - w/tauw : volt/second
                    dge/dt = -ge/taue : siemens
                    dgi/dt = -gi/taui : siemens    
                    '''
    
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='euler',
                            threshold='vm>VT',
                            reset='vm=Vr; w+=DeltaW',
                            refractory=T_ref)
                G.w = 0.0
                
            else:
                eqs = '''
                    dvm/dt  = (- gL*(vm-EL) -ge*(vm - Ee) - gi*(vm - Ei))/C : volt (unless refractory)
                    dge/dt = -ge/taue : siemens
                    dgi/dt = -gi/taui : siemens    
                    '''
    
                # neurons
                G = NeuronGroup(Nneurons_tot, eqs,method='euler',
                            threshold='vm>VT',
                            reset='vm=Vr',
                            refractory=T_ref)
    
            # intial values
            G.vm = 'Vr + rand() * (VT - Vr)'
            G.ge = 0.0
            G.gi = 0.0
            
            
        # synapses
        Se = Synapses(G[:N_e], G, 'we: siemens', on_pre='ge += we')
        Si = Synapses(G[N_e:], G, 'wi: siemens', on_pre='gi += wi')
        Se.connect('i!=j', p=pe)
        Si.connect('i+N_e!=j', p=pi)
        Se.we = "we_init*nS*rand()" # random excitatory synaptic weight (voltage)
        Si.wi = "wi_init*nS*rand()" # random inhibitory synaptic weight
        Se.delay = d*ms
        Si.delay = d*ms
        
    
    print('Number of excitatory neurons:',N_e)
    print('Number of inhibitory neurons:',N_i)
 
    
    Se.delay = d*ms
    Si.delay = d*ms
    
    
    # spikes
    spikemon = SpikeMonitor(G, record=True)
    # voltage
    statemon = StateMonitor(G, 'vm', record=np.arange(10))
    
    run(sim_t, report='text')
        
        
    # # discard first 5 seconds
    sptimes = spikemon.t/second
    idx = sptimes > 5
    spid = spikemon.i
    sptimes = sptimes[idx]
    spid = spid[idx]
    sptimes -= 5
    statetime = statemon.t/second
    statev = statemon.vm/mV

    # weights
    Jmat = np.zeros((Nneurons_tot, Nneurons_tot))
    if syn=='current':
        Jmat[Se.i,Se.j] = Se.we/mV
        Jmat[Si.i + N_e, Si.j] = Si.wi/mV
    else:
        Jmat[Se.i,Se.j] = Se.we/nS
        Jmat[Si.i + N_e, Si.j] = -Si.wi/nS

    # save data as pandas dataframe
    df = pd.DataFrame(columns=['spiketimes','spikeid','Nobs','Ntot','Jmat','Jmat_obs','obs_neurons','sim_time',
                              'adapt','type','syn','params'])
    df = df.append({'spiketimes': sptimes,
                    'spikeid': np.array(spid),
                    'Jmat': Jmat,
                    'adapt':adapt,
                    'type':case,
                    'syn':syn,
                    'params':params
                    }, ignore_index=True)
    
    filename = 'data_N'+str(Nneurons_tot)+'_T'+str(int(sim_t/second))+'_'+case+'_'+syn
    if adapt==True:
        filename = filename+'_adapt'
    print('File saved: %s' %filename)
    df.to_pickle(filename+'.pkl')
    
    # export synapses
    synapses = pd.DataFrame(columns=['ex_i','inh_i','ex_j','inh_j','ex_we','inh_wi'])
    synapses = synapses.append({'ex_i':np.array(Se.i),
               'inh_i':np.array(Si.i),
               'ex_j':np.array(Se.j),
               'inh_j':np.array(Si.j),
               'ex_we':np.array(Se.we/mV),
               'inh_wi':np.array(Si.wi/mV)}, ignore_index=True)
    synapses.to_pickle(filename+'_synapses.pkl') 
    print(np.unique(Se.j==synapses['ex_j'][0]))
