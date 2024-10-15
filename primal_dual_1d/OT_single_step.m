function [m_new,rho_new]=OT_single_step(m,rho,phi,mu,Nx,dx,Nt,dt)
    % use delta_phi to compute phi^{t-1}-phi^{t}
    delta_phi=-phi;
    for i=2:Nt 
        delta_phi(i,:)=phi(i-1,:)-phi(i,:);
    end
    a=2/mu*ones(size(m));
    b=delta_phi/dt*2+2/mu*(-rho+2*mu);
    c=delta_phi/dt*4*mu+2*mu-4*rho;
    d=-(m-mu*div_star_phi(phi,m,Nx,dx)).^2+delta_phi/dt*2*mu^2-2*mu*rho;
    rho_new=root(a,b,c,d);
    rho_new(1,:)=rho(1,:);% keep t=0 unchanged

    m_new=rho_new.*(m-mu*div_star_phi(phi,m,Nx,dx))./(rho_new+mu);
    m_new(:,end)=0;
end

