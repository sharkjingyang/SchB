function div_star_phi=div_star_phi(phi,Nx,Nt,dx)
    
    div_star_phi=zeros(Nt,Nx);
    div_star_phi(:,1)=0;
%     div_star_phi(:,1)=-phi(:,1)/dx;
    for i=2:Nx
        div_star_phi(:,i)=(phi(:,i-1)-phi(:,i))/dx;
    end
end

