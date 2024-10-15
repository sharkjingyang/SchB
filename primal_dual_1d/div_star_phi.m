function div_star_phi=div_star_phi(phi,m,Nx,dx)
    div_star_phi=zeros(size(m));
    for i=1:Nx-1
        div_star_phi(:,i)=-(phi(:,i+1)-phi(:,i))/dx;
    end
end

