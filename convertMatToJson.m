function convertMatToJson(input_path, output_path)
% CONVERTMATTOJSON convert .mat files to .json
% Give an input path and an output path and this function will convert
% all .mat functions located at the input path to json files located at
% the output file.
% Both path should contain the file extension. 
% Example:
% input_path = 'C:\Projects\ASSProject\Analysis\Data\matData\aCell1_2.mat'
% output_path = 'C:\Projects\ASSProject\Analysis\Data\jsonData\aCell1_2.json'


[output_directory_path, ~, ~] = fileparts(output_path);

if exist(output_path, 'file') == 2
    fprintf(strcat(output_path,' already exists. It will be overwritten.\n'));
elseif exist(output_directory_path,'dir') ~= 7
    fprintf(strcat('Creation of a new folder: ', output_directory_path));
    fprintf('\n');
    mkdir(output_directory_path)
end

data = load(input_path);
% Convert MATLAB structure to JSON format
json_data = jsonencode(data);
% Save the JSON data to a file
fileID = fopen(output_path, 'w');
fprintf(fileID, '%s', json_data);
fclose(fileID);

end

