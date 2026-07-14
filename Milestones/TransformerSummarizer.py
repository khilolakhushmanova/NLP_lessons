class TransformerSummarizer:
    """Amaliyot oxirida capstone moduliga ko'chiriladigan sodda wrapper."""

    def __init__(self, max_src_len=64, max_tgt_len=24, device=None):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.src_token_to_id = None
        self.src_id_to_token = None
        self.tgt_token_to_id = None
        self.tgt_id_to_token = None

    def fit(self, articles, summaries, epochs=60, lr=0.003):
        # Vocabulary
        self.src_token_to_id, self.src_id_to_token = build_vocab(articles)
        self.tgt_token_to_id, self.tgt_id_to_token = build_vocab(summaries)
    
        # Encode
        src_sequences = [
            encode_text(
                article,
                self.src_token_to_id,
                self.max_src_len,
                add_bos=False,
                add_eos=True,
            )
            for article in articles
        ]
    
        tgt_sequences = [
            encode_text(
                summary,
                self.tgt_token_to_id,
                self.max_tgt_len,
                add_bos=True,
                add_eos=True,
            )
            for summary in summaries
        ]
    
        # Batch
        src_batch = pad_sequences(src_sequences).to(self.device)
        tgt_batch = pad_sequences(tgt_sequences).to(self.device)
    
        decoder_input = tgt_batch[:, :-1]
        target_output = tgt_batch[:, 1:]
    
        # Model
        self.model = TinyTransformerSummarizer(
            src_vocab_size=len(self.src_token_to_id),
            tgt_vocab_size=len(self.tgt_token_to_id),
        ).to(self.device)
    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        # Training
        for epoch in range(1, epochs + 1):
    
            self.model.train()
    
            optimizer.zero_grad()
    
            logits = self.model(src_batch, decoder_input)
    
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_output.reshape(-1)
            )
    
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    
            optimizer.step()
    
            if epoch == 1 or epoch % 15 == 0:
                print(f"Epoch {epoch:03d} | Loss = {loss.item():.4f}")
        return self

    def summarize(self, text, max_length=20):
        self.model.eval()

        with torch.no_grad():
    
            # Source matnni encode qilish
            src = encode_text(
                text,
                self.src_token_to_id,
                self.max_src_len,
                add_bos=False,
                add_eos=True,
            )
    
            src = pad_sequences([src]).to(self.device)
    
            # Decoder <bos> bilan boshlanadi
            generated = [BOS]
    
            for _ in range(max_length):
    
                decoder_input = torch.tensor(
                    [generated],
                    dtype=torch.long,
                    device=self.device,
                )
    
                logits = self.model(src, decoder_input)
    
                # Oxirgi token uchun eng katta ehtimollikni tanlash
                next_token = logits[:, -1, :].argmax(dim=-1).item()
    
                if next_token == EOS:
                    break
    
                generated.append(next_token)

        return decode_ids(generated, self.tgt_id_to_token)

    def score(self, articles, summaries):
        scores = []

        for article, summary in zip(articles, summaries):
            prediction = self.summarize(article)
            scores.append(rouge1_f1(prediction, summary))
    
        return sum(scores) / len(scores)
            
print('tugadi')