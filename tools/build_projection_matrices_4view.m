function build_projection_matrices_4view(calibration_dir, output_mat, ncams, main_cam_idx)
% BUILD_PROJECTION_MATRICES_4VIEW Convert pairwise stereo calibration sessions
% into a multi-view projection matrix file for 3D-Primate-Tracker.
%
% The output file contains:
%     cam_mat_all: 3 x 4 x ncams projection matrix array
%
% Assumptions inherited from the original untitled.m:
%   1. Each calibrationSession_i.mat stores a MATLAB stereo calibration session.
%   2. CameraParameters1 is the primary/main camera.
%   3. CameraParameters2 is the secondary camera i.
%   4. The main camera is used as the world coordinate origin.
%
% Usage:
%   build_projection_matrices_4view('D:/project/calibration', ...
%                                  'D:/project/projection_matrices.mat', 4, 1)
%
% Notes:
%   - main_cam_idx uses MATLAB 1-based indexing.
%   - ADPT file camera-0 usually corresponds to MATLAB camera index 1.
%   - If your calibrationSession files use another naming convention, modify
%     session_file below.

if nargin < 1 || isempty(calibration_dir)
    calibration_dir = pwd;
end
if nargin < 2 || isempty(output_mat)
    output_mat = fullfile(calibration_dir, 'projection_matrices.mat');
end
if nargin < 3 || isempty(ncams)
    ncams = 4;
end
if nargin < 4 || isempty(main_cam_idx)
    main_cam_idx = 1;
end

cam_mat_all = nan(3, 4, ncams);

% Pick any non-main calibration file to read the primary camera intrinsics.
ref_cam_idx = find(1:ncams ~= main_cam_idx, 1, 'first');

for icam = 1:ncams
    if icam == main_cam_idx
        session_file = fullfile(calibration_dir, sprintf('calibrationSession_%d.mat', ref_cam_idx));
        if ~exist(session_file, 'file')
            error('Cannot find calibration file for main camera intrinsics: %s', session_file);
        end
        S = load(session_file);
        calibrationSession = S.calibrationSession;
        stereoParams = calibrationSession.CameraParameters;

        K1 = stereoParams.CameraParameters1.IntrinsicMatrix;
        cam_mat = [eye(3); [0 0 0]] * K1;
        cam_mat_all(:, :, icam) = cam_mat';
    else
        session_file = fullfile(calibration_dir, sprintf('calibrationSession_%d.mat', icam));
        if ~exist(session_file, 'file')
            error('Cannot find calibration file: %s', session_file);
        end
        S = load(session_file);
        calibrationSession = S.calibrationSession;
        stereoParams = calibrationSession.CameraParameters;

        if isempty(stereoParams.CameraParameters2.IntrinsicMatrix)
            error(['CameraParameters2.IntrinsicMatrix is empty in %s. ', ...
                   'Please check whether the Stereo Camera Calibrator session was saved correctly.'], session_file);
        end

        K2 = stereoParams.CameraParameters2.IntrinsicMatrix;
        R2 = stereoParams.RotationOfCamera2;
        T2 = stereoParams.TranslationOfCamera2;

        cam_mat = [R2; T2] * K2;
        cam_mat_all(:, :, icam) = cam_mat';
    end
end

save(output_mat, 'cam_mat_all');
fprintf('Saved projection matrices to: %s\n', output_mat);
fprintf('cam_mat_all size: %s\n', mat2str(size(cam_mat_all)));
end
