function root_record = root(a, b, c, d)
    p = -(b./a).^2/3 + c./a;
    q = 2*(b./a).^3/27 - b.*c./a.^2/3 + d./a;
    
    delta = q.^2 + 4/27*p.^3;
    root_record=delta;

    index1=(delta>0);
    u =nthroot( (-q(index1) + sqrt(delta(index1)))/2 ,3);
    v = nthroot( (-q(index1) - sqrt(delta(index1)))/2 ,3);
    z = u + v - b(index1)./a(index1)/3;
    root_record(index1)=z;

    index2=(delta<0);
    u = ((-q(index2) + 1i*sqrt(-delta(index2)))/2).^(1/3);
    z = real(u + conj(u) - b(index2)./a(index2)/3);
    root_record(index2)=z;

    index3=(delta==0);
    z = real(3*q(index3)./p(index3) - b(index3)./a(index3)/3);
    root_record(index3)=z;

% 
%     if delta > 0
%         u =nthroot( (-q + sqrt(delta))/2 ,3);
%         
%         v = nthroot( (-q - sqrt(delta))/2 ,3);
%         
%         z = u + v - b./a/3;
%         
%     elseif delta < 0
%         u = ((-q + 1i*sqrt(-delta))/2).^(1/3);
%         z = real(u + conj(u) - b./a/3);
%         
%     else
%         z = real(3*q./p - b./a/3);
%         
%     end
end