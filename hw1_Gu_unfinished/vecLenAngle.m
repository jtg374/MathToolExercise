function [length_u,length_v,angle] = vecLenAngle(u,v)
    length_u = sqrt(sum(u.^2));
    length_v = sqrt(sum(v.^2));
    if length_u*length_v ~=0
        angle = acos(dot(u,v)/(length_u*length_v));
    else
        disp('one of the vector is zero vector')
        angle = inf;
    end