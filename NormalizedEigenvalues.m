function result = NormalizedEigenvalues(T)
        a = -(T(1,1)+T(2,2)+T(3,3));
        b = T(1,1)*T(2,2)+T(1,1)*T(3,3)+T(2,2)*T(3,3)-T(1,2)*conj(T(1,2))-T(1,3)*conj(T(1,3))-T(2,3)*conj(T(2,3));
        c =-abs(det(T));
        x1 = (a^2)/3-b;
        y1 = a*b/6-c/2-(a^3)/27;
        x2 = sqrt((y1^2-(x1^3)/27));
        lamda1 = real((y1+x2)^(1/3)+(y1-x2)^(1/3)-a/3);
        x3 = (-a-lamda1)/2;
        y2 = sqrt((x3^2+c/lamda1));
        lamda2 = abs(x3+y2);
        lamda3 = abs(x3-y2);
        p1 = lamda1/(lamda1+lamda2+lamda3);
        p2 = lamda2/(lamda1+lamda2+lamda3);
        p3 = lamda3/(lamda1+lamda2+lamda3);
        
        result = [p1;p2;p3];
        result =  sort((result),'descend');
end
