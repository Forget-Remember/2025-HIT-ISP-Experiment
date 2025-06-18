function recognize_digit()
    % 手写数字识别界面
    
    % 创建图窗
    fig = figure('Name', 'Handwritten Digit Recognition', 'NumberTitle', 'off', ...
                 'Position', [300, 300, 600, 600], 'MenuBar', 'none');
    
    % 添加绘图区域
    ax = axes('Parent', fig, 'Position', [0.1, 0.2, 0.8, 0.7], 'Box', 'on', ...
              'XTick', [], 'YTick', []);
    title(ax, 'Draw a digit (0-9)');
            
    % 添加清除按钮
    uicontrol('Style', 'pushbutton', 'String', 'Clear', ...
              'Position', [100, 50, 100, 30], ...
              'Callback', @clearCanvas);
    
    % 添加识别按钮
    uicontrol('Style', 'pushbutton', 'String', 'Recognize', ...
              'Position', [400, 50, 100, 30], ...
              'Callback', @recognize);
    
    % 添加结果文本
    resultText = uicontrol('Style', 'text', 'String', 'Prediction: ', ...
                           'Position', [250, 100, 200, 30], ...
                           'FontSize', 16, 'FontWeight', 'bold');
    
    % 存储绘图数据
    data.drawing = false;
    data.points = [];
    data.W1 = [];
    data.W5 = [];
    data.Wo = [];
    
    % 设置鼠标事件
    set(fig, 'WindowButtonDownFcn', @startDrawing);
    set(fig, 'WindowButtonUpFcn', @stopDrawing);
    set(fig, 'WindowButtonMotionFcn', @draw);
    set(fig, 'UserData', data);
    
    % 加载预训练的模型权重
    if exist('MnistConv.mat', 'file')
        try
            % 加载权重变量
            loadedData = load('MnistConv.mat');
            
            % 确保权重变量存在
            if isfield(loadedData, 'W1') && isfield(loadedData, 'W5') && isfield(loadedData, 'Wo')
                data.W1 = loadedData.W1;
                data.W5 = loadedData.W5;
                data.Wo = loadedData.Wo;
                set(fig, 'UserData', data);
            else
                warndlg('Trained weights not found. Run TestMnistConv first.', 'Weights Not Found');
            end
        catch ME
            warndlg(['Error loading weights: ' ME.message], 'Load Error');
        end
    else
        warndlg('Trained weights not found. Run TestMnistConv first.', 'Weights Not Found');
    end

    function startDrawing(~, ~)
        % 开始绘图
        data = get(fig, 'UserData');
        data.drawing = true;
        set(fig, 'UserData', data);
    end

    function stopDrawing(~, ~)
        % 停止绘图
        data = get(fig, 'UserData');
        data.drawing = false;
        data.points = [data.points; nan, nan];
        set(fig, 'UserData', data);
    end

    function draw(~, ~)
        % 绘图处理
        data = get(fig, 'UserData');
        if data.drawing
            currentPoint = get(ax, 'CurrentPoint');
            x = currentPoint(1, 1);
            y = currentPoint(1, 2);
            
            if isempty(data.points)
                data.points = [x, y];
            else
                data.points = [data.points; [x, y]];
            end
            
            % 更新绘图
            plot(ax, data.points(:,1), data.points(:,2), 'k-', 'LineWidth', 20);
            axis([0 1 0 1]);
            axis off;
            
            set(fig, 'UserData', data);
        end
    end

    function clearCanvas(~, ~)
        % 清除画布
        cla(ax);
        data = get(fig, 'UserData');
        data.points = [];
        set(fig, 'UserData', data);
        set(resultText, 'String', 'Prediction: ');
    end

    function recognize(~, ~)
        % 识别绘制的数字
        data = get(fig, 'UserData');
        if isempty(data.points)
            set(resultText, 'String', 'Prediction: No digit drawn');
            return;
        end
        
        % 获取画布内容
        imgFrame = getframe(ax);
        img = rgb2gray(imgFrame.cdata);
        img = imcomplement(img);
        
        % 预处理图像
        img = imbinarize(img);
        img = imresize(img, [28, 28]);
        img = double(img);
        
        % 如果权重已加载，进行预测
        if ~isempty(data.W1) && ~isempty(data.W5) && ~isempty(data.Wo)
            % 预测数字（使用与TestMnistConv相同的步骤）
            y1 = Conv(img, data.W1);        % 卷积层
            y2 = ReLU(y1);                 % ReLU激活
            y3 = Pool(y2);                 % 池化
            y4 = reshape(y3, [], 1);       % 展平特征
            v5 = data.W5 * y4;             % 全连接层1
            y5 = ReLU(v5);                 % ReLU激活
            v = data.Wo * y5;              % 全连接层2
            y = Softmax(v);                % Softmax分类
            
            % 找到预测结果
            [~, idx] = max(y);
            prediction = idx;
            
            % 处理数字0的特殊情况（原代码中0被映射为10）
            if prediction == 10
                prediction = 0;
            end
            
            set(resultText, 'String', ['Prediction: ' num2str(prediction)]);
            
            % 显示预处理后的图像
            figure;
            subplot(1,2,1);
            imshow(imcomplement(img), 'InitialMagnification', 500);
            title('Preprocessed Digit');
            subplot(1,2,2);
            imshow(img, 'InitialMagnification', 500);
            title('Binary Digit');
        else
            set(resultText, 'String', 'Weights not loaded');
        end
    end

    % 自定义函数定义（确保这些函数在路径中）
    function y = Conv(x, W)
        % 简化的卷积函数实现
        [w, h, ~] = size(W);
        [imgW, imgH] = size(x);
        
        output = zeros(imgW - w + 1, imgH - h + 1, size(W, 3));
        
        for f = 1:size(W, 3)
            for i = 1:(imgW - w + 1)
                for j = 1:(imgH - h + 1)
                    output(i, j, f) = sum(sum(x(i:i+w-1, j:j+h-1) .* W(:, :, f)));
                end
            end
        end
        y = output;
    end

    function y = ReLU(x)
        % ReLU激活函数
        y = max(0, x);
    end

    function y = Pool(x)
        % 池化层实现
        [w, h, d] = size(x);
        y = zeros(w/2, h/2, d);
        
        for f = 1:d
            for i = 1:2:w
                for j = 1:2:h
                    if i+1 <= w && j+1 <= h
                        y(ceil(i/2), ceil(j/2), f) = max(max(x(i:i+1, j:j+1, f)));
                    end
                end
            end
        end
    end

    function y = Softmax(x)
        % Softmax函数
        ex = exp(x - max(x)); % 数值稳定处理
        y = ex / sum(ex);
    end
end