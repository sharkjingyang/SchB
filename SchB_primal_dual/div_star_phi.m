function div_star_phi=div_star_phi(phi,Nx,Nt,dx)
    
    div_star_phi=zeros(Nt,Nx);
    div_star_phi(:,1)=-phi(:,2)/dx;
    for i=2:Nx-1
        div_star_phi(:,i)=(phi(:,i-1)-phi(:,i))/dx;
    end
    div_star_phi(:,Nx)=phi(:,Nx-1)/dx;
end

