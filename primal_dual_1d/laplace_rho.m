function laplace_rho=laplace_rho(rho,m,Nx,dx)
    laplace_rho=zeros(size(m));
    laplace_rho(:,1)=(rho(:,2)-rho(:,1)-dx*m(:,1))/dx/dx;
    laplace_rho(:,Nx)=(rho(:,Nx-1)-rho(:,Nx)+dx*m(:,end))/dx/dx;
    for i=2:Nx-1
        laplace_rho(:,i)=(rho(:,i+1)-2*rho(:,i)+rho(:,i-1))/dx/dx;
    end
end