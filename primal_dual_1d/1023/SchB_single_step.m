function [m_new,rho_new]=SchB_single_step(m,rho,phi,mu,Nx,dx,Nt,dt)
    % use delta_phi to compute phi^{t-1}-phi^{t}
    delta_phi=-phi;
    for i=2:Nt 
        delta_phi(i,:)=phi(i-1,:)-phi(i,:);
    end

    laplace_phi=zeros(size(m));
    laplace_phi(:,1)=(phi(:,1)-phi(:,2))/dx/dx;
    laplace_phi(:,Nx)=(-phi(:,Nx-1)+phi(:,Nx))/dx/dx;
    for i=2:Nx-1
        laplace_phi(:,i)=(-phi(:,i-1)+2*phi(:,i)-phi(:,i+1))/dx/dx;
    end
    
    
    
    a=2/mu*ones(size(m));
    b=(delta_phi/dt+laplace_phi)*2+2/mu*(-rho+2*mu);
    c=(delta_phi/dt+laplace_phi)*4*mu+2*mu-4*rho;
    d=-(m-mu*div_star_phi(phi,Nx,Nt,dx)).^2+(delta_phi/dt+laplace_phi)*2*mu^2-2*mu*rho;
    
    rho_new=root(a,b,c,d);
    rho_new(1,:)=rho(1,:);% keep t=0 unchanged

    m_new=rho_new.*(m-mu*div_star_phi(phi,Nx,Nt,dx))./(rho_new+mu);
    m_new(:,end)=0;
  
    
end
