from evaluate_city import iou as Eval

dict = Eval.get_iou(pred=np.asarray(np.argmax(pred,axis=-1).reshape((1, -1, 1)),dtype=np.uint8),gt=np.asarray(np.argmax(gt,axis=-1).reshape((1, -1, 1)),dtype=np.uint8))
print (dict['classScores'])
print (dict['averageScoreClasses'])
