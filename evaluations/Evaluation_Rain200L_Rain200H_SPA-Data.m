% Evaluation script for Rain200L, Rain200H, and SPA-Data datasets
% This script assumes you have MATLAB installed with the Image Processing Toolbox.

% Set paths - modify according to your directory structure
RESULTS_DIR = 'results/PADUM';  % Directory containing output images
GT_DIR = 'datasets/test';       % Directory containing ground truth images

% Dataset names
DATASETS = {{'Rain200L', 'Rain200H', 'SPA-Data'}};

% Metrics to calculate
METRICS = {{'PSNR', 'SSIM'}};

for i = 1:length(DATASETS)
    dataset = DATASETS{i};
    
    % Paths for current dataset
    result_path = fullfile(RESULTS_DIR, dataset);
    gt_path = fullfile(GT_DIR, dataset, 'gt');
    
    if ~exist(result_path, 'dir')
        warning('Results directory does not exist: %s', result_path);
        continue;
    end
    
    if ~exist(gt_path, 'dir')
        warning('Ground truth directory does not exist: %s', gt_path);
        continue;
    end
    
    % Get list of image files
    result_files = dir(fullfile(result_path, '*.png'));
    result_files = {{result_files.name}};  % Extract just the filenames
    
    if isempty(result_files)
        warning('No result files found in %s', result_path);
        continue;
    end
    
    total_psnr = 0;
    total_ssim = 0;
    count = 0;
    
    % Process each file
    for j = 1:length(result_files)
        filename = result_files{j};
        
        % Skip non-image files
        [~, ~, ext] = fileparts(filename);
        if ~strcmp(ext, '.png') && ~strcmp(ext, '.jpg') && ~strcmp(ext, '.jpeg')
            continue;
        end
        
        % Full paths
        result_file = fullfile(result_path, filename);
        gt_file = fullfile(gt_path, filename);
        
        if ~exist(gt_file, 'file')
            warning('Ground truth file not found: %s', gt_file);
            continue;
        end
        
        % Read images
        result_img = imread(result_file);
        gt_img = imread(gt_file);
        
        % Ensure same size
        if ~isequal(size(result_img), size(gt_img))
            warning('Size mismatch for %s. Resizing...', filename);
            result_img = imresize(result_img, size(gt_img)(1:2));
        end
        
        % Convert to double for calculation
        result_img = im2double(result_img);
        gt_img = im2double(gt_img);
        
        % Calculate metrics
        current_psnr = calculate_psnr(result_img, gt_img);
        current_ssim = calculate_ssim(result_img, gt_img);
        
        total_psnr = total_psnr + current_psnr;
        total_ssim = total_ssim + current_ssim;
        count = count + 1;
    end
    
    if count > 0
        avg_psnr = total_psnr / count;
        avg_ssim = total_ssim / count;
        
        fprintf('Evaluation Results for %s:\n', dataset);
        fprintf('Average PSNR: %.2f dB\n', avg_psnr);
        fprintf('Average SSIM: %.4f\n\n', avg_ssim);
    else
        warning('No valid files processed for %s', dataset);
    end
end