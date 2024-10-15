Nx = 80;
Nt = 60;
mu=0.005;
tao=0.005;
N_itr=200000;
X = [-5, 5];
T = [0, 1];
dx = (X(2)- X(1)) / Nx;
dt = 1 / Nt;
x_space = linspace(X(1)+dx, X(2), Nx);
t_space = linspace(T(1), T(2), Nt + 1);
[X_plot, Y_plot] = meshgrid(x_space, t_space);
rho = rand(Nt + 1, Nx)/Nx+0.0001;
rho(1, :) = rho_0(x_space);
rho(end, :) = rho_1(x_space);

m=zeros(Nt,Nx);
phi=zeros(Nt,Nx);
rho_opt=rho(1:end-1,:);


for k=1:N_itr
    
    [m_new,rho_new]=OT_single_step(m,rho_opt,phi,mu,Nx,dx,Nt,dt);
    m_temp=2*m_new-m;
    rho_temp=2*rho_new-rho_opt;

    delta_rho=zeros(size(rho_temp));
    for t=1:Nt-1
        delta_rho(t,:)=rho_temp(t+1,:)-rho_temp(t,:);
    end
    delta_rho(Nt,:)=rho(end, :)-rho_temp(Nt,:);
    phi_new=phi+tao*(div_m(m_temp,Nx,dx)+delta_rho/dt);

    %%record gap 
    gap_m=m_new-m;
    gap_rho=rho_new-rho_opt;
    gap_phi=phi_new-phi;
    % 赋值
    m=m_new;
    rho_opt=rho_new;
    phi=phi_new;
    
    %compute gap
    R=1/mu*sum(gap_m.^2,"all")+1/mu*sum(gap_rho.^2,"all")+1/tao*sum(gap_phi.^2,"all");
    dif_rho=zeros(size(gap_rho));
    for t=1:Nt-1
        dif_rho(t,:)=gap_rho(t+1,:)-gap_rho(t,:);
    end
    R=R-2*sum(gap_phi.*(div_m(gap_m,Nx,dx)+dif_rho/dt     ),"all");
    
    % compute WFR distance
    D=1/2*sum(m.^2./rho_opt,"all");
    disp(k);
    fprintf('the value of WFR is %9.6f\n',D*dx*dt);
    fprintf('the value of gap is %9.3e\n',R);


    %test_plot
    if mod(k, 10000) == 0
       rho_plot=zeros(size(rho));
       rho_plot(1:end-1,:)=rho_opt;
       rho_plot(end,:)=rho(end,:);

       surf(X_plot, Y_plot, rho_plot);
       xlabel('x');
       ylabel('t');
       zlabel('rho');
       title('3D Plot');
       drawnow;
       pause(0.2);
       clf;
    end
end



function g = Gaussian(x, mu, sigma)
    g = 1 / sqrt(2 * pi * sigma^2) * exp(-(x - mu).^2 / (2 * sigma^2));
end

function r = rho_0(x)
    r = 1/3*Gaussian(x, -3, sqrt(1))+2/3*Gaussian(x, 3, sqrt(1));
%     r = Gaussian(x, -2, sqrt(1)) ;
end

function r = rho_1(x)
    r = Gaussian(x, 0, sqrt(1)) ;
end
