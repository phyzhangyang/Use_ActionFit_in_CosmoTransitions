def tunnelFromPhase(phases, start_phase, V, dV, Tmax,
                    Ttol=1e-3, maxiter=100, phitol=1e-8, overlapAngle=45.0,
                    nuclCriterion = lambda S,T: S/(T+1e-100) - 140.,
                    verbose = False,
                    fullTunneling_params={}):
    verbose = False
    tunnel_list = []
    highest_Tnuc = 0 # return the transiton with the highest Tnuc
    for phase in phases.values():
      ########## Get critical temperature ##########
      def DV(T): return V(start_phase.valAt(T), T) - V(phase.valAt(T), T)
      Tmin = max(start_phase.T[0],phase.T[0])
      Tmax = min(start_phase.T[-1],phase.T[-1])
      if Tmax <= Tmin:
        # no overlap
        continue
      if DV(Tmin) < 0:
        # start_phase is lower at tmin, no tunneling
        continue
      if DV(Tmax) > 0:
        # start_phase is higher even at tmax, no critical temperature
        continue
      Tcrit = optimize.brentq(DV, Tmin, Tmax, disp=False)
      ############ Get actions ####################
      Tmax = Tcrit - Ttol
      Tmin = max(Tmin,Ttol)
      if Tmax <= Tmin:
        # TODO: the temperature interval is too short
        continue
      Tlist = np.linspace(Tmin, Tmax, 20)
      T_list = []
      action_T_list = []
      for T in Tlist:
        # Get rid of the T parameter for V and dV
        def V_(x,T=T,V=V): return V(x,T)
        def dV_(x,T=T,dV=dV): return dV(x,T)
        def fmin(x): return optimize.fmin(V, x, args = (T,), xtol=phitol, ftol=np.inf, disp=False)
        x0 = fmin(start_phase.valAt(T))
        V0 = V(x0, T)
        x1 = fmin(phase.valAt(T))
        V1 = V(x1, T)
        try:
          tobj = pathDeformation.fullTunneling([x1,x0], V_, dV_, callback_data=T, **fullTunneling_params)
          action = tobj.action
        except tunneling1D.PotentialError as err:
          if err.args[1] == "no barrier":
            action = 0.0
          elif err.args[1] == "stable, not metastable":
            action = np.inf
          else:
            print(err)
            continue
        except Exception as err:
          print(err)
          continue
    
        action_T = action/T
        if action_T > 1 and action_T < 1000:
          T_list.append(T)
          action_T_list.append(action_T)
      T_list = np.array(T_list)
      action_T_list = np.array(action_T_list)
      ############ Perform fit ####################
      MSE = 100
      num_delete = 0
      while num_delete < 5:
        # fit sum a_i * T^i / (T-T_C)^2
        y = action_T_list * T_list * (T_list - Tcrit)**2
        coeff = np.polyfit(T_list, y, 5)
        polynomial = np.poly1d(coeff)
        def S_T_fun(x): return polynomial(x) / (x * (x - Tcrit)**2)
        y_fit = S_T_fun(T_list)
        residuals = y_fit - action_T_list
        # mean square error
        MSE = np.mean(residuals**2)
        if MSE < 5:
          break
        else:
          num_delete +=1
          if len(T_list)< 10 or num_delete==5:
            raise
          del action_T_list[0]
          del T_list[0]
        
      
      if verbose:
        print("S/T = ", action_T_list)
        print("T = ", T_list)
        print("residuals = ", residuals)
        print("MSE = ", MSE)
        import matplotlib.pyplot as plt
        Ti = np.linspace(Tmin,Tmax,200)
        plt.plot(Ti, S_T_fun(Ti), c="b")
        plt.scatter(T_list, action_T_list, c="r")
        plt.ylim(min(action_T_list)-10,max(action_T_list)+10)
        plt.xlabel('T')
        plt.ylabel('S/T')
        plt.savefig("S_T.png")
      
      Tmin = optimize.fmin(S_T_fun, 0.5*(Tmin+Tmax),xtol=Ttol*10, ftol=1.0, maxiter=maxiter, disp=0)[0]

      def fun_nucl(x):
        return nuclCriterion(S_T_fun(x)*x,x)
      if fun_nucl(Tmin)>0:
        # no nucleation temperature
        continue
      if fun_nucl(Tmax)<0:
        Tnuc = Tmax
      else:
        Tnuc = optimize.brentq(fun_nucl, Tmin, Tmax, xtol=Ttol, maxiter=maxiter, disp=False)
      if verbose:
        print("Tnuc = ", Tnuc)
      if highest_Tnuc > Tnuc:
        continue
      ############ Get beta/H ####################
      # u = sum a_i * T^i, v = T * (T - Tc)**2
      u = polynomial(Tnuc)
      u_prime_ = polynomial.deriv()
      u_prime = u_prime_(Tnuc)
      v = Tnuc * (Tnuc - Tcrit)**2
      v_prime = (Tnuc - Tcrit) * (3 * Tnuc - Tcrit)
      dST_dT =  (u_prime * v - u * v_prime) / (v ** 2)

      if highest_Tnuc > Tnuc:
        continue
      
      tdict = dict(low_vev=fmin(phase.valAt(Tnuc)), high_vev=fmin(start_phase.valAt(Tnuc)), Tnuc=Tnuc,  low_phase=phase.key, high_phase=start_phase.key, action=S_T_fun(Tnuc)*Tnuc, trantype=1)
      tdict['d(S/T)/dT'] = dST_dT
      tunnel_list.append(tdict)
      
    return tdict if tunnel_list else None
