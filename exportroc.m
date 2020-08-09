function exportroc(x,net,t,tr)
    % Training Confusion Plot Variables
    yTrn = net(x(:,tr.trainInd));
    tTrn = t(:,tr.trainInd);
    % Validation Confusion Plot Variables
    yVal = net(x(:,tr.valInd));
    tVal = t(:,tr.valInd);
    % Test Confusion Plot Variables
    yTst = net(x(:,tr.testInd));
    tTst = t(:,tr.testInd);
    % Overall Confusion Plot Variables
    yAll = net(x);
    tAll = t;
    % Plot Confusion
    figure
    plotroc(tTrn, yTrn, 'Training', tVal, yVal, 'Validation', tTst, yTst, 'Test', tAll, yAll, 'Overall')
end
