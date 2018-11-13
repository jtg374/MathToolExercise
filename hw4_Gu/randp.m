function sample = randp(pdf,num)
    cdf = cumsum(pdf); % cumulitive probability function, set partition points between [0 1)
    sample = rand(num,1); % uniform random numbers
    for ii = 1:num
        sample(ii) = sum(cdf<sample(ii)); % transform according to partition points
    end