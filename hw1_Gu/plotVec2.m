function plotVec2(M)
    if size(M,1) == 2
        figure; hold on
        for ii = 1:size(M,2)
            plot([0,M(1,ii)],[0,M(2,ii)],'-b','Marker','o')
        end
        plot([-1,1],[0,0],'k')
        plot([0,0],[-1,1],'k')
        hold off
        axis equal
        xlim([-1,1])
        ylim([-1,1])
    else
        disp('wrong matrix size, row number should be 2, instead of')
        disp(size(M,1))
    end
        