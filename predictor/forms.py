from django import forms

class BusinessForm(forms.Form):
    STATE_CHOICES = [
        ('AL', 'Alabama'), ('AK', 'Alaska'), ('AZ', 'Arizona'), ('AR', 'Arkansas'), ('CA', 'California'),
        ('CO', 'Colorado'), ('CT', 'Connecticut'), ('DE', 'Delaware'), ('FL', 'Florida'), ('GA', 'Georgia'),
        ('HI', 'Hawaii'), ('ID', 'Idaho'), ('IL', 'Illinois'), ('IN', 'Indiana'), ('IA', 'Iowa'),
        ('KS', 'Kansas'), ('KY', 'Kentucky'), ('LA', 'Louisiana'), ('ME', 'Maine'), ('MD', 'Maryland'),
        ('MA', 'Massachusetts'), ('MI', 'Michigan'), ('MN', 'Minnesota'), ('MS', 'Mississippi'), ('MO', 'Missouri'),
        ('MT', 'Montana'), ('NE', 'Nebraska'), ('NV', 'Nevada'), ('NH', 'New Hampshire'), ('NJ', 'New Jersey'),
        ('NM', 'New Mexico'), ('NY', 'New York'), ('NC', 'North Carolina'), ('ND', 'North Dakota'),
        ('OH', 'Ohio'), ('OK', 'Oklahoma'), ('OR', 'Oregon'), ('PA', 'Pennsylvania'), ('RI', 'Rhode Island'),
        ('SC', 'South Carolina'), ('SD', 'South Dakota'), ('TN', 'Tennessee'), ('TX', 'Texas'),
        ('UT', 'Utah'), ('VT', 'Vermont'), ('VA', 'Virginia'), ('WA', 'Washington'),
        ('WV', 'West Virginia'), ('WI', 'Wisconsin'), ('WY', 'Wyoming')
    ]
    CATEGORIES_CHOICES = [
        ('restaurant', 'Restaurant'), ('health', 'Health'), # Define based on your categories
        ('auto', 'Automotive'), ('beauty', 'Beauty & Spas'),
    ]

    state = forms.ChoiceField(choices=STATE_CHOICES, label='State')
    categories = forms.ChoiceField(choices=CATEGORIES_CHOICES, label='Business Categories')
    accepts_credit_cards = forms.BooleanField(required=False, label='Accepts Credit Cards')
    outdoor_seating = forms.BooleanField(required=False, label='Outdoor Seating')
    open_on_weekends = forms.BooleanField(required=False, label='Open on Weekends')
    total_weekly_hours = forms.IntegerField(min_value=0, label='Total Weekly Hours')
   


class BusinessPredictionForm(forms.Form):
    text = forms.CharField(widget=forms.HiddenInput(), initial="Default review text", label="Review Text")
    total_hours_week = forms.FloatField(label="Total Hours Open Per Week")
    is_weekend_open = forms.ChoiceField(choices=((0, "No"), (1, "Yes")), label="Open on Weekends")
    state = forms.CharField(max_length=100, label="State")
    categories = forms.CharField(max_length=300, label="Categories")