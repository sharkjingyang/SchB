function div_m = div_m(m,Nx,dx)
    div_m=zeros(size(m));
    div_m(:,1)=m(:,2)/dx;
    div_m(:,end)=-m(:,end)/dx;
    for i=2:Nx-1
        div_m(:,i)=(m(:,i+1)-m(:,i))/dx;
    end
end

