function symmetry = computeSymmetrySimple(mask, direction)
    % Calcula simetria simple
    [rows, cols] = size(mask);
    
    if strcmp(direction, 'horizontal')
        half = floor(cols/2);
        left = mask(:, 1:half);
        right = fliplr(mask(:, end-half+1:end));
        symmetry = sum(left(:) & right(:)) / min(sum(left(:)), sum(right(:)));
    else
        half = floor(rows/2);
        top = mask(1:half, :);
        bottom = flipud(mask(end-half+1:end, :));
        symmetry = sum(top(:) & bottom(:)) / min(sum(top(:)), sum(bottom(:)));
    end
    
    if isnan(symmetry)
        symmetry = 0;
    end
end