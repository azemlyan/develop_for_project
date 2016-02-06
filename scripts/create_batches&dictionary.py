import artm.messages_pb2
import artm.library
cpc = artm.messages_pb2.CollectionParserConfig()
cpc.vocab_file_path = '/home/andrew/develop_for_diplom/data/vocab.kos.txt'
cpc.docword_file_path = '/home/andrew/develop_for_diplom/data/docword.kos.txt'
cpc.target_folder = '/home/andrew/develop_for_diplom/batch'
artm.library.Library().ParseCollection(cpc)
cpc.dictionary_file_name = 'dictionary'
cpc.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci
artm.library.Library().ParseCollection(cpc)
with artm.library.MasterComponent() as master:
	master.ImportDictionary('dictionary', '/home/andrew/develop_for_diplom/batch/dictionary')
	perplexity_score = master.CreatePerplexityScore()
	sparsity_theta_score = master.CreateSparsityThetaScore()
	sparsity_phi_score = master.CreateSparsityPhiScore()
	top_tokens_score = master.CreateTopTokensScore()
	theta_snippet_score = master.CreateThetaSnippetScore()
	smsp_theta_reg = master.CreateSmoothSparseThetaRegularizer()
	smsp_phi_reg = master.CreateSmoothSparsePhiRegularizer()
	decorrelator_reg = master.CreateDecorrelatorPhiRegularizer()
	model = master.CreateModel(topics_count=5, inner_iterations_count=5)
	model.EnableRegularizer(smsp_theta_reg, -0.1)
	model.EnableRegularizer(smsp_phi_reg, -0.1)
	model.EnableRegularizer(decorrelator_reg, 100)
	model.Initialize('dictionary')   
	for iteration in range(0, 5):
		master.InvokeIteration(disk_path='/home/andrew/develop_for_diplom/batch/dictionary')  
		master.WaitIdle()                                # and wait until it completes.
		model.Synchronize()                              # Synchronize topic model.
		print "Iter#" + str(iteration),
		print ": Perplexity = %.3f" % perplexity_score.GetValue(model).value,
		print ", Phi sparsity = %.3f" % sparsity_phi_score.GetValue(model).value,
		print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value

    	artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
    	artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))
