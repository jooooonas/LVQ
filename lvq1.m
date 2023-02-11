function [] = lvq1(P, N, dataset, labels, K, n, t_max)
    % initialize K random prototypes per class
    prototypes = init_protos(P, N, dataset, labels, K);

    % initialize array to store E (the number of missclassified data
    % points) in percent
    E_array = zeros(t_max, 1);

    % iterate through all epochs
    for t = 1:t_max
        % set a random order for each epoch
        iterating_order = randperm(P);
        
        %iterate through the whole data set
        for i = 1:P
            datapoint = dataset(iterating_order(i),:); % get next datapoint
    
            % get the winning prototype
            [index_proto] = get_nearest_prototype(prototypes, datapoint);
    
            % update winner
            if prototypes(index_proto,3) == labels(iterating_order(i))
                % if the winner's class and the datapoint's class are the
                % same -> move closer
                prototypes(index_proto,1:2) = prototypes(index_proto,1:2) + n * (datapoint - prototypes(index_proto,1:2));
            else
                % if the winner's class is not the same as the datapoint's
                % class -> move away
                prototypes(index_proto,1:2) = prototypes(index_proto,1:2) - n * (datapoint - prototypes(index_proto,1:2));
            end
        end
        % store E value
        E_array(t) = missclassified_examples(P, dataset, labels, prototypes) / 100;
    end
    % plot the final positions of prototypes together with the data set
    plot_protos(P, prototypes, dataset);
    figure()
    x_axis = linspace(1, t_max, t_max);
    plot(x_axis, E_array); % plot E as a function of the number of epochs
    xlabel('Epochs, t');
    ylabel('Training error in %');
    title(sprintf('Training error for K = %d, \\eta = %0.3f', K, n));
end

function [E] = missclassified_examples(P, dataset, labels, prototypes)
    E = 0; % initialize return value with 0
    
    % iterate over data set and count the misslabeled data points
    for i = 1:P
        % get winner
        index_proto = get_nearest_prototype(prototypes, dataset(i,:));
        % check labelling
        if prototypes(index_proto, 3) ~= labels(i)
            E = E + 1;
        end
    end
end

function [index_proto] = get_nearest_prototype(prototypes, datapoint)
    % set first prototype as closest
    distance = pdist2(prototypes(1,1:2), datapoint);
    index_proto = 1;
    
    % check if any other prototype is closer to datapoint than the first one
    for i = 2:length(prototypes(:,1))
        dist_tmp = pdist2(prototypes(i,1:2), datapoint);
        
        % if closer: set index to new winner and update distance
        if (dist_tmp < distance)
            distance = dist_tmp;
            index_proto = i;
        end
    end
end

function [prototypes] = init_protos(P, N, dataset, labels, K)
    number_classes = length(unique(labels)); % get the number of different classes
    prototypes = zeros( number_classes * K, N + 1); % initialize matrix for prototypes

    rng('default'); % create a seed for reproducibility

    % iterate through the different classes
    for i = 0:number_classes - 1
        % get random indices of the prototypes
        prototypes_indices = randperm(P / number_classes, K);

        % get the corresponding datapoints to the indices
        for j = 1:K
            prototypes(j + (i * K), 1:N) = dataset((P/number_classes) * i + prototypes_indices(j), :);
            prototypes(j + (i * K), N + 1) = i + 1; % set class label in extra column
        end
    end
end

function [] = plot_protos(P, prototypes, dataset)
    labels = string.empty;
    % create labels for plotting
    for i = 1:P
        if (i <= 50)
            labels(i) = 'class 1';
        else
            labels(i) = 'class 2';
        end
    end
    for i = 1:length(prototypes(:,1))
        % label different prototypes differently
        if prototypes(i,3) == 1
            labels(i + P) = 'prototype class 1';
        else
            labels(i + P) = 'prototype class 2';
        end
    end

    % concatenate dataset-matrix and prototypes-matrix
    result = cat(1, dataset, prototypes(:,1:2));

    %plotting
    figure()
    h = gscatter(result(:,1), result(:,2), labels');
    h(3).Color = h(1).Color;
    h(4).Color = h(2).Color;
    h(3).Marker = '*';
    h(4).Marker = '*';
    hold on
end