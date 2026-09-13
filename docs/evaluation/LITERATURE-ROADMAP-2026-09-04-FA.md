# نقشه ادبیات برای بهترکردن مدل PrivChain-MDD

تاریخ جست‌وجو: 2026-09-04  
وضعیت: مرور اولیه منابع اصلی؛ پیش از پیاده‌سازی نهایی باید جدول استخراج نظام‌مند تکمیل شود.

## سؤال این مرور

چه تغییری احتمالاً مدل تشخیص افسردگی را بهتر می‌کند، بدون آنکه عدد از نشت سوژه، promptهای Ellie یا انتخاب مکرر روی Dev حاصل شود؟

## نتیجه اجرایی کوتاه

اولویت پیشنهادی بر اساس شواهد فعلی:

1. افزودن شاخه مستقل **Conversational Temporal Dynamics (CTD)**.
2. کامل‌کردن فیلتر صوت به‌صورت یک ablation مستقل، نه تغییر مستقیم baseline.
3. حفظ participant-only بودن ورودی‌های اصلی.
4. حفظ ارزیابی کاملاً subject-disjoint و دست‌نخورده نگه‌داشتن Test.
5. بررسی speaker-disentanglement به‌عنوان آزمایش بعدی حریم خصوصی/utility.
6. استفاده از E-DAIC یا داده بیشتر برای افزایش توان آماری، در صورت دسترسی قانونی.

## ۱. promptهای Ellie و diagnostic shortcut

مقاله Burdisso و همکاران نشان می‌دهد مدل می‌تواند از promptهای مصاحبه‌گر، مخصوصاً ناحیه سؤال‌های مربوط به سابقه سلامت روان، به‌عنوان میان‌بر تشخیصی استفاده کند. آن‌ها با بهره‌برداری عمدی از این bias به F1 برابر 0.90 رسیده‌اند. مدل participant-only شواهد را از بخش گسترده‌تری از مصاحبه جمع می‌کند.

منبع اصلی:

- Burdisso et al., 2024, *DAIC-WOZ: On the Validity of Using the Therapist's Prompts in Automatic Depression Detection from Clinical Interviews*: https://arxiv.org/abs/2404.14463
- نسخه منتشرشده در ClinicalNLP/NAACL 2024: https://doi.org/10.18653/v1/2024.clinicalnlp-1.8

نتیجه برای پروژه:

- متن Ellie نباید وارد text encoder شود.
- صدای Ellie نباید وارد acoustic depression encoder شود.
- زمان پایان سؤال Ellie را می‌توان فقط برای ساخت ویژگی interactional مانند response latency استفاده کرد. این استفاده باید صریحاً از استفاده محتوایی تفکیک شود.
- participant-only بودن یک تصمیم اعتبارسنجی است؛ کاهش احتمالی F1 در مقایسه با روش‌های آلوده، دلیل برگشتن به promptهای Ellie نیست.

## ۲. Conversational Temporal Dynamics

Kang و همکاران یک بردار ۲۴بعدی از زمان‌بندی تعامل را با WavLM-large و RoBERTa-large مقایسه کرده‌اند. CTD بهترین modality منفرد روی Development بوده است. late fusion وزن‌دار به macro-F1 برابر 0.804 روی Development و 0.669 روی Test رسیده و fusion سه‌گانه در راه‌حل گزارش‌شده به acoustic وزن صفر داده است.

منبع اصلی:

- Kang et al., 2026, *Can Conversational Temporal Dynamics Improve Depression Detection in Dyads?*: https://arxiv.org/abs/2607.03744

محدودیت منبع:

- نسخه فعلی preprint و submitted to SLT 2026 است.
- خود نویسندگان یافته را preliminary و وابسته به یک benchmark کوچک معرفی می‌کنند.
- عدد بالای Development نباید به‌جای نتیجه Test نقل شود.

نتیجه برای پروژه:

CTD بالاترین اولویت مهندسی دارد، چون ویژگی‌ها کم‌بعد، تفسیرپذیر و ارزان‌اند. شاخه پیشنهادی باید از transcript timestamps و VUV خام ساخته شود و جدا از acoustic encoder باقی بماند.

ویژگی‌های کاندید:

- تعداد نوبت‌های participant و Ellie
- میانگین، میانه، انحراف معیار، چارک‌ها و بیشینه مدت پاسخ participant
- latency از پایان نوبت Ellie تا شروع پاسخ participant (نسخه اولیه کد تا
  زمان افزودن parser دوگوینده‌ای از فاصله پاسخ‌های متوالی به‌عنوان proxy استفاده می‌کند)
- نسبت کل زمان participant، Ellie، سکوت و overlap
- تعداد و طول مکث‌های داخل نوبت با استفاده از VUV
- voiced ratio و silence-to-speech ratio
- نرخ کلمه/توکن بر ثانیه
- تعداد پاسخ‌های تک‌کلمه‌ای و پاسخ‌های کمتر از یک آستانه زمانی
- backchannel count، فقط پس از تعریف واژگانی ثابت و ثبت‌شده
- missingness indicators برای ویژگی‌هایی که در همه جلسه‌ها قابل محاسبه نیستند

## ۳. فیلتر صوت و نوبت‌های کوتاه

منابع مختلف participant speech را با timestamps جدا می‌کنند و سکوت پس‌زمینه را کنار می‌گذارند. مطالعه speaker-leakage سال ۲۰۲۶، participant-only speech را استخراج و هر پنج utterance متوالی را در یک segment ادغام می‌کند. یک پیاده‌سازی عمومی DAIC-WOZ نیز پاسخ‌های کمتر از یک ثانیه را برای استخراج VGGish حذف می‌کند؛ اما این repository به‌تنهایی استاندارد علمی یا اثبات بهبود طبقه‌بندی نیست.

منابع:

- Yeh et al., 2026, preprocessing و split کنترل‌شده speaker overlap: https://arxiv.org/abs/2604.14354
- نمونه پیاده‌سازی threshold یک ثانیه: https://github.com/LIU70KG/daic_woz_processing_master

نتیجه محتاطانه:

- حذف `VUV=0` برای ورودی acoustic یک فرضیه معقول است، اما هنوز از مرور فعلی نمی‌توان آن را «استاندارد قطعی سه‌مرحله‌ای» نامید.
- حذف turnهای زیر یک ثانیه ممکن است برای embeddingهای نیازمند حداقل طول ضروری باشد، ولی می‌تواند پاسخ‌های کوتاه و بالینی مهم را نیز حذف کند.
- بنابراین A2 باید ablation باشد و نه پیش‌فرض بدون آزمون.
- VUV خام و turnهای کوتاه باید برای CTD حفظ شوند، حتی اگر از acoustic encoder حذف شوند.

## ۴. نشت هویت گوینده

مطالعه کنترل‌شده Yeh و همکاران نشان می‌دهد overlap گوینده می‌تواند عملکرد را به‌شدت بالا ببرد. برای نمونه، accuracy یک Wav2Vec fine-tuned از 58.74% در حالت speaker-independent به 97.65% در حالت overlap رسیده است. مدل overlapped هم‌زمان speaker-ID accuracy بسیار بالایی داشته است. DANN شکاف را کاملاً حذف نکرده است.

منبع اصلی:

- Yeh et al., 2026: https://arxiv.org/abs/2604.14354

نتیجه برای پروژه:

- split باید در سطح participant کاملاً جدا باشد؛ پروژه فعلی این شرط را رعایت می‌کند.
- segmentهای یک participant هرگز نباید میان train و evaluation پخش شوند.
- افزایش معماری به‌تنهایی مشکل هویت را حل نمی‌کند.
- speaker probe و re-identification که امروز اجرا شدند باید به‌عنوان کنترل اعتبار حفظ شوند.

نکته تصحیحی:

این مقاله `session normalization` را به‌عنوان راه‌حل اصلی توصیه نمی‌کند. راه‌حل‌های صریح آن split مستقل از گوینده و آزمایش DANN هستند. بنابراین دفاع از session normalization باید به ablation خود پروژه یا منابع دیگری متکی باشد، نه به این مقاله.

## ۵. speaker disentanglement

Ravi و همکاران adversarial speaker disentanglement را بررسی کرده‌اند. در مطالعه ۲۰۲۲، کاهش speaker-discriminability همراه با بهبود depression-discriminability گزارش شده و Wav2Vec2 به F1 برابر 69.2% روی DAIC-WOZ رسیده است. مطالعه بعدی نیز پیوند utility و voice privacy را بررسی کرده است.

منابع اصلی:

- Ravi et al., 2022: https://arxiv.org/abs/2206.09530
- Ravi et al., Computer Speech & Language 2024: https://doi.org/10.1016/j.csl.2024.101605

نتیجه برای پروژه:

پس از CTD و A2، یک بازوی gradient-reversal speaker head می‌تواند هم‌زمان با معیار depression و re-identification سنجیده شود. این تغییر پرریسک‌تر و پرهزینه‌تر از CTD است و اولویت دوم محسوب می‌شود.

## ۶. اندازه واقعی عملکرد foundation modelها

مطالعه Interspeech 2025، HuBERT و RoBERTa را روی DAIC+ و DEPTALK مقایسه کرده است. صفحه رسمی مقاله برای DAIC+ حدود F1=0.70 برای متن و F1=0.75 برای مدل چندوجهی fine-tuned for emotion recognition گزارش می‌کند؛ صوت در دو dataset حدود 0.60 است.

منبع اصلی:

- Gómez-Zaragozá et al., Interspeech 2025: https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.html

تصحیح عدد:

اعداد `HuBERT=0.553`، `RoBERTa=0.595` و multimodal برابر `0.643` از صفحه رسمی و abstract این مقاله تأیید نشدند. ممکن است مربوط به یک جدول، split یا variant خاص باشند؛ تا استخراج مستقیم جدول PDF نباید به مقاله نسبت داده شوند.

## ۷. بی‌ثباتی رتبه‌بندی روی dataset کوچک

Multi-Probe Audit تحت LOSO subject-disjoint یک مرجع محافظه‌کارانه macro-F1 برابر 0.723 گزارش می‌کند. همچنین نشان می‌دهد رتبه روش‌ها روی validation و official test می‌تواند ناپایدار باشد و top-3 overlap صفر شود.

منبع اصلی:

- Ishikawa & Duke, 2026: https://arxiv.org/abs/2605.23977

نتیجه برای پروژه:

- انتخاب روش با یک اجرای Dev قابل دفاع نیست.
- multi-seed، paired bootstrap و Test دست‌نخورده ضروری‌اند.
- اگر توان محاسباتی اجازه دهد، nested CV یا LOSO مکمل نتیجه رسمی Train/Test باشد.

## ۸. ویژگی‌های دینامیک صوتی فراتر از میانگین

یک preprint سال ۲۰۲۶ گزارش می‌کند entropy و trajectory dynamics از static pooling بهتر بوده‌اند و برای entropy biomarkers آزمون permutation معنادار ارائه می‌دهد.

منبع:

- Samanta, 2026, *Entropy-Dominated Temporal Vocal Dynamics as Digital Biomarkers for Depression Detection*: https://arxiv.org/abs/2604.26998

نتیجه برای پروژه:

اگر CTD مفید بود، مرحله بعد می‌تواند entropy و تغییرات زمانی voiced ratio، pitch و energy باشد. به‌دلیل جدید و تک‌مطالعه‌ای بودن منبع، این مسیر پس از CTD ساده قرار می‌گیرد.

## برنامه آزمایش پیشنهادی

بدون تغییر split، seedها یا metric اصلی، چهار بازوی نخست اجرا شوند:

| بازو | Acoustic filtering | CTD |
|---|---|---|
| B0 | baseline فعلی | خاموش |
| B1 | حذف VUV=0 و turn کوتاه | خاموش |
| B2 | baseline فعلی | روشن |
| B3 | حذف VUV=0 و turn کوتاه | روشن |

قواعد:

- ویژگی‌های CTD فقط با آمار Train استاندارد شوند.
- Ellie فقط در timestamp interaction استفاده شود، نه در embedding محتوایی.
- VUV خام پیش از فیلتر برای CTD محاسبه شود.
- predictionهای report ذخیره و اختلاف AUC به‌صورت paired bootstrap محاسبه شود.
- F1، ROC-AUC، PR-AUC، sensitivity و specificity گزارش شوند.
- علاوه بر utility، re-identification هر بازو دوباره سنجیده شود.
- ابتدا smoke، سپس ۵ seed برای غربال و تنها در صورت سیگنال از پیش تعریف‌شده ۱۰ seed اجرا شود.

## تصمیم فعلی

بهترین تغییر کم‌هزینه و دارای پشتوانه، **CTD به‌عنوان شاخه مستقل** است. A2 باید هم‌زمان ولی جداگانه آزمایش شود. رفتن مستقیم به مدل بنیادی بزرگ‌تر قبل از حل زمان‌بندی، leakage و توان آماری توصیه نمی‌شود.
