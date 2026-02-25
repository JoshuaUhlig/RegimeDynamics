using Random
using DelimitedFiles

function powerlawrv(alpha, tau0)
    """
    samples a powerlaw distribution  1/t^{1+alpha}
    # Arguments
    - `tau0`: smallest time to be drawn
    - `alpha`: -(1+alpha) is the scaling of the power law 
    """
    return tau0/(1-rand())^(1/alpha)
end

function powerlawrv_cutoff(alpha,tau0,tau1)
    """
    samples a powerlaw distribution 1/t^{1+alpha} with cutoff
    # Arguments
    - `tau0`: smallest time to be drawn
    - 'tau1': largest time to be drawn
    - `alpha`: -(1+alpha) is the scaling of the power law 
    """
    return 1/(rand()*(tau1^(-alpha)-tau0^(-alpha))+tau0^(-alpha))^(1/alpha)
end


function vdem_powerlaw_ctrw(T,alpha,x0,y0,res=1.0,alphax1=-0.785,alphax2=0.575,alphay1=-0.785,alphay2=0.64,cutoffxy1=10^(-6),cutoffx2=0.03,cutoffy2=0.03,cutoffx3=11.16444072945315,cutoffy3=5.525608279496182)
    """
    Simulates 2D CTRW 
    # Arguments
    - `T`: runtime of simulation
    - `res`: resolution, smallest available time step
    - `sigma`: standard deviation of jump size distr in direction i
    - `alpha`: -(1+alpha) is the exponent of the waiting time powerlaw

    """
    N = Int(T/res)+1
    t = range(0,T,step=res)
    r = zeros((N,2))
    r[1,:]=[x0,y0]
    xcurr = x0
    ycurr = y0
    twait = 0.0
    step = 2
    Bx = 1/(cutoffx2^(alphax1-alphax2)*(cutoffxy1^(-alphax1)-cutoffx2^(-alphax1))/alphax1+(cutoffx2^(-alphax2)-cutoffx3^(-alphax2))/alphax2)
    Ax = cutoffx2^(alphax1-alphax2)*Bx
    By = 1/(cutoffy2^(alphay1-alphay2)*(cutoffxy1^(-alphay1)-cutoffy2^(-alphay1))/alphay1+(cutoffy2^(-alphay2)-cutoffy3^(-alphay2))/alphay2)
    Ay = cutoffy2^(alphay1-alphay2)*By
    
    probxin1 =  Ax*(cutoffxy1^(-alphax1)-cutoffx2^(-alphax1))/alphax1
    probyin1 =  Ay*(cutoffxy1^(-alphay1)-cutoffy2^(-alphay1))/alphay1
    dx,dy = 0,0
    while  step <= N 
        twait = twait + res*round(1.0/res*powerlawrv(alpha,res))
        reachedend= step>N
        while t[step]<=twait
            r[step,:]=[xcurr,ycurr]
            step=step+1
            reachedend= step>N
            if reachedend break end
        end
        # PC1
        if reachedend break end
        if rand()<probxin1
            dx = powerlawrv_cutoff(alphax1,cutoffxy1,cutoffx2)
        else dx = powerlawrv_cutoff(alphax2,cutoffx2,cutoffx3)
        end
        if rand()>0.5
            xcurr = xcurr +dx 
        else xcurr = xcurr -dx
        end
        # PC2
        if rand()<probyin1
            dy = powerlawrv_cutoff(alphay1,cutoffxy1,cutoffy2)
        else dy = powerlawrv_cutoff(alphay2,cutoffy2,cutoffy3)   
        end     
        if rand()>0.5
            ycurr = ycurr +dy
        else ycurr = ycurr-dy
        end
        r[step,:]=[xcurr,ycurr]
        step = step+1
    end
    return r,vcat(t)

end

function writetofile(name,data,dir="./",acc=14)
    open(dir*name*".dat","w") do file
        writedlm(file, round.(data,digits=acc),'\t')
    end
end

function sim_ctrw_ens(ens_size,T,alpha,res=1.0,dir="./")
    x0= 0.0
    y0= 0.0
    for i=1:ens_size
        r,t = vdem_powerlaw_ctrw(T,alpha,x0,y0)
        writetofile(string(i),r,dir)
    end
end



#major parameters
ens_size=10000
T=30
res =1.0
#Potential parameters and powerlaw
alpha=1.038 


println("Simulating with parameters: alpha="*string(alpha))
sim_ctrw_ens(ens_size,T,alpha,res,"sim_trajectories_lb/")
