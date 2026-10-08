# CISI test collection

Downloaded on October 8, 2026 from the University of Glasgow IR test collection
archive:

    http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/cisi.tar.gz

The same collection is mirrored on Kaggle as "CISI (a dataset for Information
Retrieval)". It was produced at the Centre for Inventions and Scientific
Information and distributed with the SMART retrieval system.

| File     | Contents                                                      |
|----------|---------------------------------------------------------------|
| CISI.ALL | 1,460 documents: number (.I), title (.T), authors (.A), abstract (.W), citations (.X) |
| CISI.QRY | 112 queries. 1-76 are user questions; 77-112 are abstracts used as queries |
| CISI.REL | 3,114 binary relevance judgements covering 76 of the queries   |
| CISI.BLN | Boolean versions of the first queries (not used by this project) |

The files are unmodified. `cisi_ir_system.py` parses them directly.
