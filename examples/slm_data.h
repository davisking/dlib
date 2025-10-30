#ifndef SLM_DATA_H
#define SLM_DATA_H

#include <string>
#include <vector>
#include <algorithm>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>

// Utility function to concatenate text parts
inline std::string concatenateTexts(const std::vector<std::string>& texts) {
    std::string result;
    for (const auto& text : texts) {
        result += text;
    }
    return result;
}

// Text parts for training
// Used by the <slm_basic_train_ex.cpp> example
const std::vector<std::string> shakespeare_text_parts = {
    R"(QUEEN ELIZABETH:
Send to her, by the man that slew her brothers,
A pair of bleeding-hearts; thereon engrave
Edward and York; then haply she will weep:
Therefore present to her--as sometime Margaret
Did to thy father, steep'd in Rutland's blood,--
A handkerchief; which, say to her, did drain
The purple sap from her sweet brother's body
And bid her dry her weeping eyes therewith.
If this inducement force her not to love,
Send her a story of thy noble acts;
Tell her thou madest away her uncle Clarence,
Her uncle Rivers; yea, and, for her sake,
Madest quick conveyance with her good aunt Anne.

KING RICHARD III:
Come, come, you mock me; this is not the way
To win our daughter.

QUEEN ELIZABETH:
There is no other way
Unless thou couldst put on some other shape,
And not be Richard that hath done all this.

KING RICHARD III:
Say that I did all this for love of her.

QUEEN ELIZABETH:
Nay, then indeed she cannot choose but hate thee,
Having bought love with such a bloody spoil.

KING RICHARD III:
Look, what is done cannot be now amended:
Men shall deal unadvisedly sometimes,
Which after hours give leisure to repent.
If I did take the kingdom from your sons,
To make amends, Ill give it to your daughter.
If I have kill'd the issue of your womb,
To quicken your increase, I will beget
Mine issue of your blood upon your daughter
A grandam's name is little less in love
Than is the doting title of a mother;
They are as children but one step below,
Even of your mettle, of your very blood;
Of an one pain, save for a night of groans
Endured of her, for whom you bid like sorrow.
Your children were vexation to your youth,
But mine shall be a comfort to your age.
The loss you have is but a son being king,
And by that loss your daughter is made queen.
I cannot make you what amends I would,
Therefore accept such kindness as I can.
Dorset your son, that with a fearful soul
Leads discontented steps in foreign soil,
This fair alliance quickly shall call home
To high promotions and great dignity:
The king, that calls your beauteous daughter wife.
Familiarly shall call thy Dorset brother;
Again shall you be mother to a king,
And all the ruins of distressful times
Repair'd with double riches of content.
What! we have many goodly days to see:
The liquid drops of tears that you have shed
Shall come again, transform'd to orient pearl,
Advantaging their loan with interest
Of ten times double gain of happiness.
Go, then my mother, to thy daughter go
Make bold her bashful years with your experience;
Prepare her ears to hear a wooer's tale
Put in her tender heart the aspiring flame
Of golden sovereignty; acquaint the princess
With the sweet silent hours of marriage joys
And when this arm of mine hath chastised
The petty rebel, dull-brain'd Buckingham,
Bound with triumphant garlands will I come
And lead thy daughter to a conqueror's bed;
To whom I will retail my conquest won,
And she shall be sole victress, Caesar's Caesar.

QUEEN ELIZABETH:
What were I best to say? her father's brother
Would be her lord? or shall I say, her uncle?
Or, he that slew her brothers and her uncles?
Under what title shall I woo for thee,
That God, the law, my honour and her love,
Can make seem pleasing to her tender years?

KING RICHARD III:
Infer fair England's peace by this alliance.

QUEEN ELIZABETH:
Which she shall purchase with still lasting war.

KING RICHARD III:
Say that the king, which may command, entreats.

QUEEN ELIZABETH:
That at her hands which the king's King forbids.

KING RICHARD III:
Say, she shall be a high and mighty queen.

QUEEN ELIZABETH:
To wail the tide, as her mother doth.

KING RICHARD III:
Say, I will love her everlastingly.

QUEEN ELIZABETH:
But how long shall that title 'ever' last?

KING RICHARD III:
Sweetly in force unto her fair life's end.

QUEEN ELIZABETH:
But how long fairly shall her sweet lie last?

KING RICHARD III:
So long as heaven and nature lengthens it.

QUEEN ELIZABETH:
So long as hell and Richard likes of it.

KING RICHARD III:
Say, I, her sovereign, am her subject love.

QUEEN ELIZABETH:
But she, your subject, loathes such sovereignty.

KING RICHARD III:
Be eloquent in my behalf to her.

QUEEN ELIZABETH:
An honest tale speeds best being plainly told.

KING RICHARD III:
Then in plain terms tell her my loving tale.

QUEEN ELIZABETH:
Plain and not honest is too harsh a style.

)",

    R"(KING RICHARD III:
Your reasons are too shallow and too quick.

QUEEN ELIZABETH:
O no, my reasons are too deep and dead;
Too deep and dead, poor infants, in their grave.

KING RICHARD III:
Harp not on that string, madam; that is past.

QUEEN ELIZABETH:
Harp on it still shall I till heart-strings break.

KING RICHARD III:
Now, by my George, my garter, and my crown,--

QUEEN ELIZABETH:
Profaned, dishonour'd, and the third usurp'd.

KING RICHARD III:
I swear--

QUEEN ELIZABETH:
By nothing; for this is no oath:
The George, profaned, hath lost his holy honour;
The garter, blemish'd, pawn'd his knightly virtue;
The crown, usurp'd, disgraced his kingly glory.
if something thou wilt swear to be believed,
Swear then by something that thou hast not wrong'd.

KING RICHARD III:
Now, by the world--

QUEEN ELIZABETH:
'Tis full of thy foul wrongs.

KING RICHARD III:
My father's death--

QUEEN ELIZABETH:
Thy life hath that dishonour'd.

KING RICHARD III:
Then, by myself--

QUEEN ELIZABETH:
Thyself thyself misusest.

KING RICHARD III:
Why then, by God--

QUEEN ELIZABETH:
God's wrong is most of all.
If thou hadst fear'd to break an oath by Him,
The unity the king thy brother made
Had not been broken, nor my brother slain:
If thou hadst fear'd to break an oath by Him,
The imperial metal, circling now thy brow,
Had graced the tender temples of my child,
And both the princes had been breathing here,
Which now, two tender playfellows to dust,
Thy broken faith hath made a prey for worms.
What canst thou swear by now?

KING RICHARD III:
The time to come.

QUEEN ELIZABETH:
That thou hast wronged in the time o'erpast;
For I myself have many tears to wash
Hereafter time, for time past wrong'd by thee.
The children live, whose parents thou hast
slaughter'd,
Ungovern'd youth, to wail it in their age;
The parents live, whose children thou hast butcher'd,
Old wither'd plants, to wail it with their age.
Swear not by time to come; for that thou hast
Misused ere used, by time misused o'erpast.

KING RICHARD III:
As I intend to prosper and repent,
So thrive I in my dangerous attempt
Of hostile arms! myself myself confound!
Heaven and fortune bar me happy hours!
Day, yield me not thy light; nor, night, thy rest!
Be opposite all planets of good luck
To my proceedings, if, with pure heart's love,
Immaculate devotion, holy thoughts,
I tender not thy beauteous princely daughter!
In her consists my happiness and thine;
Without her, follows to this land and me,
To thee, herself, and many a Christian soul,
Death, desolation, ruin and decay:
It cannot be avoided but by this;
It will not be avoided but by this.
Therefore, good mother,--I must can you so--
Be the attorney of my love to her:
Plead what I will be, not what I have been;
Not my deserts, but what I will deserve:
Urge the necessity and state of times,
And be not peevish-fond in great designs.

QUEEN ELIZABETH:
Shall I be tempted of the devil thus?

KING RICHARD III:
Ay, if the devil tempt thee to do good.

QUEEN ELIZABETH:
Shall I forget myself to be myself?

KING RICHARD III:
Ay, if yourself's remembrance wrong yourself.

)",

    R"(QUEEN ELIZABETH:
But thou didst kill my children.

KING RICHARD III:
But in your daughter's womb I bury them:
Where in that nest of spicery they shall breed
Selves of themselves, to your recomforture.

QUEEN ELIZABETH:
Shall I go win my daughter to thy will?

KING RICHARD III:
And be a happy mother by the deed.

QUEEN ELIZABETH:
I go. Write to me very shortly.
And you shall understand from me her mind.

KING RICHARD III:
Bear her my true love's kiss; and so, farewell.
Relenting fool, and shallow, changing woman!
How now! what news?

RATCLIFF:
My gracious sovereign, on the western coast
Rideth a puissant navy; to the shore
Throng many doubtful hollow-hearted friends,
Unarm'd, and unresolved to beat them back:
'Tis thought that Richmond is their admiral;
And there they hull, expecting but the aid
Of Buckingham to welcome them ashore.

KING RICHARD III:
Some light-foot friend post to the Duke of Norfolk:
Ratcliff, thyself, or Catesby; where is he?

CATESBY:
Here, my lord.

KING RICHARD III:
Fly to the duke:
Post thou to Salisbury
When thou comest thither--
Dull, unmindful villain,
Why stand'st thou still, and go'st not to the duke?

CATESBY:
First, mighty sovereign, let me know your mind,
What from your grace I shall deliver to him.

KING RICHARD III:
O, true, good Catesby: bid him levy straight
The greatest strength and power he can make,
And meet me presently at Salisbury.

CATESBY:
I go.

RATCLIFF:
What is't your highness' pleasure I shall do at
Salisbury?

KING RICHARD III:
Why, what wouldst thou do there before I go?

RATCLIFF:
Your highness told me I should post before.

KING RICHARD III:
My mind is changed, sir, my mind is changed.
How now, what news with you?

STANLEY:
None good, my lord, to please you with the hearing;
Nor none so bad, but it may well be told.

KING RICHARD III:
Hoyday, a riddle! neither good nor bad!
Why dost thou run so many mile about,
When thou mayst tell thy tale a nearer way?
Once more, what news?

STANLEY:
Richmond is on the seas.

KING RICHARD III:
There let him sink, and be the seas on him!
White-liver'd runagate, what doth he there?

STANLEY:
I know not, mighty sovereign, but by guess.

KING RICHARD III:
Well, sir, as you guess, as you guess?

STANLEY:
Stirr'd up by Dorset, Buckingham, and Ely,
He makes for England, there to claim the crown.

KING RICHARD III:
Is the chair empty? is the sword unsway'd?
Is the king dead? the empire unpossess'd?
What heir of York is there alive but we?
And who is England's king but great York's heir?
Then, tell me, what doth he upon the sea?

STANLEY:
Unless for that, my liege, I cannot guess.

KING RICHARD III:
Unless for that he comes to be your liege,
You cannot guess wherefore the Welshman comes.
Thou wilt revolt, and fly to him, I fear.

STANLEY:
No, mighty liege; therefore mistrust me not.

KING RICHARD III:
Where is thy power, then, to beat him back?
Where are thy tenants and thy followers?
Are they not now upon the western shore.
Safe-conducting the rebels from their ships!

STANLEY:
No, my good lord, my friends are in the north.

KING RICHARD III:
Cold friends to Richard: what do they in the north,
When they should serve their sovereign in the west?

)",

    R"(STANLEY:
They have not been commanded, mighty sovereign:
Please it your majesty to give me leave,
I'll muster up my friends, and meet your grace
Where and what time your majesty shall please.

KING RICHARD III:
Ay, ay. thou wouldst be gone to join with Richmond:
I will not trust you, sir.

STANLEY:
Most mighty sovereign,
You have no cause to hold my friendship doubtful:
I never was nor never will be false.

KING RICHARD III:
Well,
Go muster men; but, hear you, leave behind
Your son, George Stanley: look your faith be firm.
Or else his head's assurance is but frail.

STANLEY:
So deal with him as I prove true to you.

Messenger:
My gracious sovereign, now in Devonshire,
As I by friends am well advertised,
Sir Edward Courtney, and the haughty prelate
Bishop of Exeter, his brother there,
With many more confederates, are in arms.

Second Messenger:
My liege, in Kent the Guildfords are in arms;
And every hour more competitors
Flock to their aid, and still their power increaseth.

Third Messenger:
My lord, the army of the Duke of Buckingham--

KING RICHARD III:
Out on you, owls! nothing but songs of death?
Take that, until thou bring me better news.

Third Messenger:
The news I have to tell your majesty
Is, that by sudden floods and fall of waters,
Buckingham's army is dispersed and scatter'd;
And he himself wander'd away alone,
No man knows whither.

KING RICHARD III:
I cry thee mercy:
There is my purse to cure that blow of thine.
Hath any well-advised friend proclaim'd
Reward to him that brings the traitor in?

Third Messenger:
Such proclamation hath been made, my liege.

Fourth Messenger:
Sir Thomas Lovel and Lord Marquis Dorset,
'Tis said, my liege, in Yorkshire are in arms.
Yet this good comfort bring I to your grace,
The Breton navy is dispersed by tempest:
Richmond, in Yorkshire, sent out a boat
Unto the shore, to ask those on the banks
If they were his assistants, yea or no;
Who answer'd him, they came from Buckingham.
Upon his party: he, mistrusting them,
Hoisted sail and made away for Brittany.

KING RICHARD III:
March on, march on, since we are up in arms;
If not to fight with foreign enemies,
Yet to beat down these rebels here at home.

CATESBY:
My liege, the Duke of Buckingham is taken;
That is the best news: that the Earl of Richmond
Is with a mighty power landed at Milford,
Is colder tidings, yet they must be told.

KING RICHARD III:
Away towards Salisbury! while we reason here,
A royal battle might be won and lost
Some one take order Buckingham be brought
To Salisbury; the rest march on with me.

DERBY:
Sir Christopher, tell Richmond this from me:
That in the sty of this most bloody boar
My son George Stanley is frank'd up in hold:
If I revolt, off goes young George's head;
The fear of that withholds my present aid.
But, tell me, where is princely Richmond now?

CHRISTOPHER:
At Pembroke, or at Harford-west, in Wales.

DERBY:
What men of name resort to him?

CHRISTOPHER:
Sir Walter Herbert, a renowned soldier;
Sir Gilbert Talbot, Sir William Stanley;
Oxford, redoubted Pembroke, Sir James Blunt,
And Rice ap Thomas with a valiant crew;
And many more of noble fame and worth:
And towards London they do bend their course,
If by the way they be not fought withal.

DERBY:
Return unto thy lord; commend me to him:
Tell him the queen hath heartily consented
He shall espouse Elizabeth her daughter.
These letters will resolve him of my mind. Farewell.

BUCKINGHAM:
Will not King Richard let me speak with him?

Sheriff:
No, my good lord; therefore be patient.

BUCKINGHAM:
Hastings, and Edward's children, Rivers, Grey,
Holy King Henry, and thy fair son Edward,
Vaughan, and all that have miscarried
By underhand corrupted foul injustice,
If that your moody discontented souls
Do through the clouds behold this present hour,
Even for revenge mock my destruction!
This is All-Souls' day, fellows, is it not?

Sheriff:
It is, my lord.

BUCKINGHAM:
Why, then All-Souls' day is my body's doomsday.
This is the day that, in King Edward's time,
I wish't might fall on me, when I was found
False to his children or his wife's allies
This is the day wherein I wish'd to fall
By the false faith of him I trusted most;
This, this All-Souls' day to my fearful soul
Is the determined respite of my wrongs:
That high All-Seer that I dallied with
Hath turn'd my feigned prayer on my head
And given in earnest what I begg'd in jest.
Thus doth he force the swords of wicked men
To turn their own points on their masters' bosoms:
Now Margaret's curse is fallen upon my head;
'When he,' quoth she, 'shall split thy heart with sorrow,
Remember Margaret was a prophetess.'
Come, sirs, convey me to the block of shame;
Wrong hath but wrong, and blame the due of blame.

)"
};

// Full text for training
const std::string shakespeare_text = concatenateTexts(shakespeare_text_parts);

// Prompt for text generation
const std::string shakespeare_prompt = R"(QUEEN ELIZABETH:
But thou didst kill my children.

KING RICHARD III:
But in your daughter's womb I bury them:
Where in that nest of spicery they shall breed
Selves of themselves, to your recomforture.

QUEEN ELIZABETH:
Shall I go win my daughter to thy will?

KING RICHARD III:
And be a happy mother by the deed.

QUEEN ELIZABETH:
I go. Write to me very shortly.
And you shall understand from me her mind.

)";

// Retrieves the contents of an internal "plain text" file
// Used by the <slm_advanced_train_ex.cpp> example
const std::string get_internal_data_file()
{
    dlib::base64 base64_coder;
    dlib::compress_stream::kernel_1ea compressor;
    std::ostringstream sout;
    std::istringstream sin;

    sout << "QT8JNRyIvUkLem5lyDM28N2jHsZ8QbVJftbxUAb+POOMDyYijyrFN3tmZ2bGDEr2sp+RI6hYVcrs";
    sout << "iWZ6IuTswZDqPQfxnObLRcOjNf8volT05DrVOEV3jGYnVKakok2oU1ptoKozH3fVhj3L6+Gmd17D";
    sout << "MsgzBDH2xF74KyhsPmvn7PICSGEeipD4kSf28nRGUZbqW+C0zbVjFH5+ct0An3x70dH6orb51ZG4";
    sout << "5n+A2170qip5ScyTUcaT1jI2vxS5pNkRa7bvg4ambdY1F2w3rdgKqiJZMHhP6BKfE3f5YNhpWTRc";
    sout << "Wr8komzNMVxD3URfSsNnzsgk8XKVbzsAvxWE9hr/6M/ObA2WviOEqh6Kk1xtSGyodq9r7xfaYUyf";
    sout << "YDz5BSqZzsL1Zp+MqE0s/BhFBG/QYU0MbV9x6v2qvkLJQndkEOSfeMqPU6mItZahqnWhRi2cLRbU";
    sout << "wRflvZPttqxoT+8TijkhYWs3hTdJs7I8fIJGKN58eY8QPt5j3OmLFFmJux2Cs+CIJWaNs0l1nJgb";
    sout << "/rSbDR2ksyR7fr1n5qxwKpVhHzWxnHLbXwkChZ4q7bFKKXgoH7AnHz3wu2ToES7yfv34rQLO9bY1";
    sout << "AXMetkdqntJ+AETV9UvWzpOHYOxGocZfWw5Fyz+6txoWJ+5iroOy1IoA2KBonIqG8g5F6LfWgwSp";
    sout << "gaFkWQrBgWp+Ump4PnA5OP+/OJAP2TF3eVbr34N/BO7tLFmrQi6ukPM3CfmE6FmflNqaCvgyLs2G";
    sout << "DOJY8kylAp779fCWBpd/kTm7BDBTPOMLWYAcT6OK02EAcanlV88Ldd4DOM7pLP3PhPZBBNhl/MYZ";
    sout << "5tOkpwV4Ivmz57C97Q6TzMAU2/gBtjAKteIYYbvMl7xdPI2toImavLI9MpPYXJw0TGHymAacJ+yF";
    sout << "FpD7kjTxwQF+ILgRSjTH8At54cbI4Y2Gyf+mCVr9qNvOTOvnDo0uIlxCiLK2oilOriV70QVMa8pU";
    sout << "dVnew4z32w0MATHYjQgWg/TFXpimLbQst/lRKhW0+l5KdVe/P/jPIwB7UR3MQPg3T8p8WIuXB+TX";
    sout << "VKEF+vJN2NJ1icNpA/4I9X/o8DMPXnl70r5HAFIiCulQ8mqkYaDIACOCX1xq/sGLUXmElWTxgars";
    sout << "EfXzjQTsgATtaARWatOasbTZh7SBH4CCpRnB7fYfIDlB/LJbVPsW4fhOxh5FST92QBUmjKSFNWwd";
    sout << "BMv6JNqY1h516Yd3WG1lOoFdvDPD0dOnhZL1sOy2jya4o4nIFPQ5+TEtjbsf06Phi9cEYeYdf409";
    sout << "pNpY6losWJOZUiraf5QFv3rbPryHMx+Zj2z+FHgZ+vsJurMvlHNgGCzpS26azM0l4Ed86g0qd7lc";
    sout << "nBl40tu+4s3GoZulbscBuofA9wjjRc012TTq40/08o43m/ihTeFM/KoGFlOf2v60OIsq27eab61F";
    sout << "OsJ1vKycYLZXEzCvMdN66fgetGkTPzF9b5hRILG1PpBtHIJH/+FDUoFVzUJznYWT8EZ7qdW0F5+9";
    sout << "WlXsL5gQJ7PV4ffSjWYHbw3R50xcsbU8m56VlufWW+L0jhSoCEahYn05HYBGf8D7fDspkGF24La4";
    sout << "MJhjOSs6r2c4ICgLo9X5Bfncee9nsXLTmi+QdKszy0TmQbEpTrhtP2wzpgMleUHHxMlbin7MP5YW";
    sout << "4MITFI53rh06whYd/IbwE7siF4JV7iFNPtT9qu7VC10MqPtIbm8AFxRQfgINAXcokh62bkqVEBEM";
    sout << "o+H1Lzoji6+TalslrSiyjgYmqQcdtP+47KWPwnOorGud1dYfQu6yplRWhHhmGWUAHKrR4j8XhSbz";
    sout << "gd5xgfm3kAj1WiIjYWJ5cuZws+KUnQVqDs+3v6w6oDn//pty2Xk4ogVXHVqqzCwWGNmr30ywPjMR";
    sout << "ce15ZPue6TQZ+hRRaq4RN4m6CZdX0nM684xZNvRsG/JmCbzTrUacjvhBWNOjrbg3PyqrFFIGfnIX";
    sout << "CRWNJPur+r0tlERGA54kW+8EofT7XwM8co1NdCoPhthnioJNnkEYE7OEhPu3OVwjLTmtC1/fxg4w";
    sout << "sAeKkrPaPkJWl/rvnTFBZz7iVbAYkMBkps4EwxfzT02hY4H9aWmFqHXhGbuglimDAdBTTr+1FLgJ";
    sout << "N5kII9wy3x6zSib8T7UGxgLEHp5Z06LZflvCp1PzAYu/m6EzPGx2mKV9zoxeCcP2KLXjXt/k5BQs";
    sout << "JJvxMg+2cr5ztZOeGmGH5l256HPB6PAX7CqKae5EWgJsVNpk5l5zBiCAL+zZ6ff8bDM0MXlYm7U8";
    sout << "LvaWOafIAXQEmTsEtsu6Cta/l57M5VxrNLtUgEUB1g/mLiCxZK5CoVhB/O38bGn1los3BJRNUAuA";
    sout << "f9aUMfyJdg4GcdD6LqhGhiMlgPNcXNRDw+gcpZWbcbBrzJEDEJwxa0unaRCPz3ofVBGuBGcNVOaC";
    sout << "sRcYWqCud1X1uhteo2wAsQa2PL1OG334n/EK/gn+fmQ++gZtj9hQBZB+OWPu8Ne61J/vuS21AumA";
    sout << "art4I6UhX05mIlfX9jXqkiRKNyzYmrWDXuKXxRYPsXfA71gmjIix3DIRhOm4ck60LkyRgHX4Rd1Z";
    sout << "WNEdXGWZLuA5supwGBxP84JkIqP6dJlgG7J+uJJkxWRUrrmuPhJtAKqc2wuk1hQhNh8uPCovj4r7";
    sout << "8PQ7T1y3Qb4A4It6we95qScZ+CYxJJuRMn2CTFqWGVXQG4IGSAYYdxrsvhq+COlInIPuuMhLNOWH";
    sout << "A9AnVXoKA2d6Ai00DGCH+up7prj62JAv+OIZ4MyskajttAQ78pT7N+JfCMAlVPnN5+QUEqIVVpbN";
    sout << "X2hLZ66Zelxq3NQlxRpcoKfM642bbZdgtIywsRF0tgJw6EkD5HRobz2KqdtS49XjH2j2JwN+wRnC";
    sout << "2UJJP45pv3Q6usif9O/FsKZuoLlsZHQQyVZe7+mmvdCY2MGFV/zs4PF3Vsg5wJuNmEnEw2FIiR3Y";
    sout << "iXal6mZcIIqwi6S7X4KMOF4a210Oij8CpUfMHKvH44tNKavWUO+em4Jr6r/Q9WqKZP2ypv50igVq";
    sout << "V/Y22dL4RkzlfVAH8/ZzrhlXB0fdFK5zqWiN5wN2oiMSeUtQ/znRLCCNlM8WE57zaZM+eruc5jxy";
    sout << "dwh0PoidlnqIkrjLntDBnT6yGKDsDieDBv9WLQ0AZVUQXINHnoQAO2dgpzB46KaJNYDJ+ZH3djFW";
    sout << "LMvuP6s2Ac83ruEJ3NeB03a/neqViQx34PMYKX/zmOvf7DXVj5QVao56CXp1ITOnviEo31PTQ5s4";
    sout << "pjUlrtEcMwhzvW+yOTLfDLWiUKNOV4MNrVEoBFcGN+X85pHluOFDBSARKaT8vQmVWKcSow7lOGsy";
    sout << "vLj236mqpUdpDvIHylonTjtnD+JGu1PWJflKEVXB/YDOh8ASovudwgAhCJE1enKNIUcngdRXYDUZ";
    sout << "d2KNal8mYOmRti3aBCGWv9BFKYn3dLMCP3BAS0V7VXf4tDVAKYOWljOFvOoJv7IGKhfHeLQURaJ7";
    sout << "RFtx3+4KBXEgBOKDeGu9/XnSkY4Dj9m2sFmjXT9/NRcLZZWOxYbXmD47QjVaV0cUW0G4iJ6xnAnv";
    sout << "SS+13UFj3xbtlMrIz7Na/B2yTKiw0QTJbSXZ3lSHbELvglDfpUP5re/bb90e3G+IViHdtQKEsaKA";
    sout << "R71aO4wM6zM6avPomduTxBsd5WpQoHSvvzSNwGzN3C2ATxKXx4XcizZAyjsOqY2oHFS3g+PmRnmp";
    sout << "we06v0Ea/CX+dZ7Y2Pmx0oGJ6FEgYL7MBPKhJerfjLtEeS9VQe5zPrhO7FpuEqZ1MUV0MO3MaOSF";
    sout << "Ytn9L63Gc91NTrvcocHnJ6faZ8f7va13+kJVSr9/Y9+07g6LcMezeJKDW0fQ//6JahNi1/VFfP2a";
    sout << "/G0qFGJrx0hO64NubdS4IcAnmVsyXhxKP1igo3QhlXcw5ZFATcfljYlbh8vkQ5MXRc6+7BILEGRy";
    sout << "n533RXrIDTAO9tTG8V4G5zqq57X4yMmbh+ALtI9Xteqk4RvcrKXb5PjXeqx9GwL4t0OMLD+cF41n";
    sout << "su8x7IDh62AdXf7EkIXLjActcdpnsFIR12nzvyhxs+flpCWSCyMwpqy7ASaO/0yYttcTJNNhvlC5";
    sout << "zp0WgOfvzPu9/CnnARrsYnkJaKSrZAgLvxQf6FZv4ECnQ6yyJIWZOG3BQ3YrWCNYHbHn1hfWaKl2";
    sout << "esmRMpsNqrlTiIihNwL1ySzJ+oZT9H1M11iqqdUUIyIpDSDOBxN8VyKM38ykqQpl+kCO5rSon7fT";
    sout << "sagqrJR5wCfXHOUY55ixmf32gq5iq00Ih6WRjLGpHRiSl/dhbAGzxYKSgrBeGZKaTWKpt7BPATcl";
    sout << "nWLYJsPQYdqSncyLJ3fNzAS1gIoxuAxqbuEpRYobX4wXFZBEn6rn1I9wnivjRgpt+BNOBj2QcFgw";
    sout << "aJLinc1pcGFaY/mLhH4I9PTYq903a3w4tuw0xALDL/cQKjvbOPXp/lhl67Eyi6ZVOVbkJB34ccKE";
    sout << "wjWG7LU45b1v2WtzDVfOhu17GM1rk1q9VWKVftDixnIDDeUjx12wsJRZ5V0Jnxs1/ajWrzLbmnND";
    sout << "EBsiloE1ng3yajyhH4qL9VL/7r4n8MwXlQanpCsIwmiaydKbnjKTxfGMqSeEwHSq0sA0k+hSJXxW";
    sout << "hDLdgRMdmk/22Z8grtkpJvhU8tamcEPD6QT2xxZP2AunCnsI0FyP/MhFWIfxxFbAxqAAMEZKzl+U";
    sout << "AjBusrUfEKjTwFCXMzQ63ArMVDl96hmRvQYMmDiBddaoS0QZqaljSWOWO7Syyqht4p8kTZ9zMQ1S";
    sout << "7ntDdB63y7GEIOutDpm6LP0Tu+piROxhdrmrf3w3s2u6XmMwhXntIFxbxes04EsqFxQfK6xmb3ws";
    sout << "NZiidakdtSbOj2QOistt7agZz0JVwe6N863TOVID6FS2fp/73u7mOn5yTTKjFjp83pOf6cQADCzw";
    sout << "GWdhc5eULE7U9Tx8JSCn7/X48hKbtQT9LS/WcH7lRN96eWG1p+uSLe4Bu5LnW8MvosxIJXdAhNRy";
    sout << "ct6XLuT5CZt5rGITmKICiKyFdrtPpueLyFIQ0Q3jByNWNrtFo6aQENMSmGM3hQgQfVJNHiKajM9t";
    sout << "ugsXtQb29TK1SpjyvY4sHLaEiXAPywbHQ14eVcEWa/sjqOavrnrTtWjIEvL2XiZ6KHZYf7yDx6Xq";
    sout << "NibtkmOtiVxlFXuk1Y52iZ4UUuJft5liEo/jFUX67RHY6xFUqk/tCpzeSL/hME5qw7cx55ROGTcg";
    sout << "QqLzD3U5G2QqtfBXokVzsXgmY9H4O40VBJi84Im0tWHyhVe/p6oLzLeRevjcDfmuNXM7o3veJajU";
    sout << "Vu/8X+Zng/m4JF55LWDKpf6ArsxweoThhTmdtrhLjhzkHABhYaZXEd93+jGoY+iKClS2B1/2AOqP";
    sout << "V5O7zO35eoU99xzIDC6+ZglcGhxFn6uVc13oOwqyQJgojvT7LGtHOsBxpAz8j3+E9S8WDmMzUkEl";
    sout << "3J4tO5gZZKVi3qrNqXKs9SowG+mGid170UnPrlp+9XeYorlNdwNVbkOM8/gBDgIZ28LFega2nFtX";
    sout << "/D8ckqTz2E6bkU6yljZ8rzpCmrHB++8h5W3izJPgsw2JpyTxgeGvq/R6JSqlhccADAc7qHsFBT9c";
    sout << "W37yC0S0TPQu1E202GQ6dI5a6IPFvhVfhZ+XwByWBm+qyI5BkXFhd7aDXiPJ/6i72InsuQUr6YV6";
    sout << "qS33wlXMdT69KjNDmJvNMY6H3WWdW7uawPpt4TJCwzGZsw6o8Z51q5Gj1l7mSmENiZUIFBYe/uku";
    sout << "AE07Z/K5DsamWDrCzB/HByaLCSouElNvRgD0dy74YSuir5o6gkWMlgtLif9Q12tg9a2VN4aEbRGz";
    sout << "WqiR2REmklVmiSHy6Eqp9kKvtoKHECIaWWXk2i3qqZ79xaBrYuh66Qtq7TTnnoeZSY0APDewwMsX";
    sout << "7J0O1sOd7pzxBKoCUpmccPgcOmLzUNU28j1Hz4ehbJX2F1kU6F+IL3kZlOLk+U4Z5Llwz757AS/4";
    sout << "IuQqtqkfgnLco88sFTlfSezhuMyO0q1d3GFPuqIOBEi5bEBGQG7niZPGfdPeE5al4drM8iMWcXpF";
    sout << "M1L4Rv7B1BrIhmj+a1tSRYF9j/kNfvW7m7bXjhfeJ4FgSTzrM7F9Q5PAqPeN9UEGHtSviyAiWcqG";
    sout << "p/ttcZfEcHXqaMZ/SUmhvCasHVEZQTtqaojjIdDh55pOLd/HYQZEiiqZumwSWirmaKEnEFpYIvmn";
    sout << "gHhvZ03jEoMbkDec3/hCfyKf9iboM0Or6VZyZ5emKv/Gvyz/YEUYOGvWIkyMWJ5SwnEFSMF1Hhvx";
    sout << "GKZ+VUv7D7yEa/XMcM7llUJ7mPW2ErQa1qM5eLp5kinFDtEw63xYG+nAg8XAT2pss1js/pprv05P";
    sout << "53Bu5GaxNw8TiTgJTT9LCRWyc5XmuB+KudTz27gkB+vuX9LQgjwRZxRFZjF0WhvDZZGE8LHCcYuF";
    sout << "EdIMbihlEeNxJq+0mbyzno5eDQIwl5FdCl+i4cD7dUhhuMVeNFjMuPxEdbXX3RQ0EXkVphZX3N6R";
    sout << "3iJJ7H7oLfQhe7W23+UmvCaCx0VLqfdwWt9l+R2mrUItecivLKjSVptHUk3kAK1nTnjBCFFPMhKk";
    sout << "gVynIOzQp+xasG0OW+3rZxR97TDWnbFj7kl7qY4tGxeln+xfouCRWJCpjTpKlZ45sJH1zSpkqmoh";
    sout << "xBv9TGmxahVA8flkcKrYP6VdsmY58o4EN1mO0eZKRiZi8kSoDGihIf/agcYNN6gTzJdxYLQVcKjc";
    sout << "69KGvZOM8o5ERaPM9Mi5+bctSKCMb9e2OlP1aYPum/84y4IvjAXVK0jecnvwZgNivF//4rDrmNNS";
    sout << "r+YwTr9wsg4B1fak6GdkwWPSjEnHX1R+Sn+X3i42DF0YJ3AwSB6my5hcXhBo7odoUYeTpSmGTjNo";
    sout << "N4KKogzsCKnyTIIa2mK+jdtqiDLWJB0jcHnusL3kGRDM6Kd+XhJk4nG6mA5xRFoYuu2IfGkAIwRK";
    sout << "xlSl9RKix6/VL/OgRIrLLO3Loi6tFvOikTqS7AUMiPzRs0ZHarsx65ThhAHw8u95R+7iSZd+juTZ";
    sout << "KHjlCgOCXxhTUqkXNc+dQh6coqRI6+Zr8+ETCzRQJMxdY4RFgyYFpAiQf5fYxf11CMZArd0X/L6C";
    sout << "0fjkI0ICHiAJGT1/qzaqqoTEFDybcue3EmNv/ZtkE+4nLjX7It8za9cGTSvRGeorOL0fCMEO/Pa+";
    sout << "/cbY6/D7QdO/cY4unu3CJIB1JmrDL4VV5vo1WaglQHbtpnv+r4+HWO7KLf8dm/f2VcAoIJ4NOvGa";
    sout << "tGMuiuLJn3Yj0jh/+k/rjgxVDo2yqkJSIUFIwJ9Zd0JK7I9f2vtQwJw0C0grk70rdivsqheSxWWW";
    sout << "ot/TyryX9Nmsm+igbSzGs2rer4DJb2T4H/vm50cnGOHVs5ttYD9N/SNLb75xXk7FRWV8TXvOSuyh";
    sout << "QU7Ntc6qDLktj8n4Zc3bXdtuEPNDkWPdJQvcOizwyrMjjd8zZeLWTjWGZtHl/85Km+kob8uYqEmS";
    sout << "8pejfi/NuCyvZ3KGRKX90qu6ZRvGLhqMIjwFzzuR1PSSe6SuJkYALMyI5LfDQXoHQ1ksxwFQVLy7";
    sout << "PHv10rr88rxRbmr6EyqeSrKdGoS6Dee/tpq+8lZsf/eeKb7k/ZhcZEfNumSf7WPsjMgkatsXkikn";
    sout << "B7CL4IgDWgUWIhE8FS6h3ZK5D2HNeDbzD0jT6iKW7MpT+XcbvcajqPkIs+UsMbL5X9Jrss/b1MkI";
    sout << "I+1GSj8emDxZA9xUbhjBzX5zGFBBxHzI6FZutmMMdgDRMMwuhpQ9OSgTvTMxK7TnyHVoDcfoHd+4";
    sout << "e9IywxCOUcKfehSF/AA8U9mSK1QRVd8wEefq04K+iOmwGrnxzKooa3MsFt95/9Fty5CVO6DJ9ABi";
    sout << "bseoMNu6sAN1rwOBez/v4+u/gXh95MFq/Ps3kLNVOscPACMztm2FpRR+4D1UKJtiw4aMcPLVw2U4";
    sout << "Re9fg6yUttVnT/2Zpi40DYYBWwcEpu9jYqNXfVvpPnaFEyYyqyuvlO3xAeY9xOU9LBt11wnxqAU/";
    sout << "d3IsXqMRW/fg3aD0li1uWBMXOtxltWHZiwehDFmQuEvIO801I8DsuZHniGKnVGvs18gwT3+xw79V";
    sout << "JooLIlUFAB3qPtsMPlgXBlRrjsp4aPZReLwFz5thoI4pXDDjr0FoXfpja8SR8y/2A1JAYkhBoQn1";
    sout << "eCbp+Lj0GCks1q/BKck3CLJYmfGlEdNRnKbureBdyS9FP+QXivlcYHGX9I3tNE9aaJwGj+x1pXgJ";
    sout << "FDCHwZS/rI4Z+MYuaDKNPv0gnMAqozvpW7V55nXQn1UpYzfq6bMEB1jVaivfHtFsPaAs5NH3lPNJ";
    sout << "KTvp9lyce++JND5ucXEy4gbU6KIqjydn92fKiVGitPo2+cdksINDjNuZ0kp/FNpaJZ7eTEzlsoQL";
    sout << "1Yuvj23499OP/Ic2eczI0sHb/3etqRtiqlGSJt0tEmviwugXesu/W6h90rw4B6SNJieKtI0BjJdn";
    sout << "XA7/IdIkFJBhbP+bUWbmyVP3e1gkrumx37i4pk8vUTc4mUgmgEiUtQT5jPPoxLP4Gv/XkihKDJfn";
    sout << "39F/u4D4GwK0E2SQCTgvzdFox9RF82Jcr8mqTG4ZioN0eUiztSRKSeCDstFuio61MMXVHZYP7vPR";
    sout << "UOqHO0JEjHvVxBaliN0rcr3OIGyXnbNMQp7SqLOX6fxgvXTmiBvKDyKrD/uDDuc1h5JhlHr6oqHi";
    sout << "lRJb9cjWQL5+I4bV4OJcwUsXihSo0aIjgTtgN55MiKlh2tpzyuHo5uJ5v4Yjc/nGAZb5vgt/nK2z";
    sout << "+4RsfCJ2YEWM5x9/RfShO0RB11qXesVlzCYiI7n9a9theAMWx8GbQNMRFepAkLroxrbhi+r8kBi7";
    sout << "aD830HnkdsZ+J/xARtgP+hP4AVy62+YTTZmMwmL2o7NsYmnJFEm3StJBfYlVhxvYfi7XksYnh8aC";
    sout << "1dpxGlAgnlVkJYrByDKRDN8pl71AfXjco0qEFWX7qNOvnwrIjw6tbhXNwJvnUaltkh928POn2du4";
    sout << "fScR7d8o3zwKxJsl4ToTArOJ4iMwBKtQm246acDAk6kUigELMmU5/AWN7SbHgmtyGAfhSlp6fj/9";
    sout << "aq8xRTAeptxyHYVsSQw+23SwGnKLIYscVE3Z8nf6rVff3doYKofr0sgEUKoC6f5ZrIftCg2jWZjh";
    sout << "QPOR5DtQToLK4qDikasT4MzxumifQgNETkprdBrknTO0FgX9/u6dglySnKuHa3LSTTOg/zN9Wpyq";
    sout << "pR5lJVWAhj7GDCBePEFdDT76sP7wn8gMzqlECOXNNMjKPINC/T2Y9YxCT2g4N8IEnm4L1cpaEjqF";
    sout << "idiTgGd1jGhNhC7jLKMzSaC2ZVesNP+kl90zuRTphhQ584qPe3UsvcD9SwQqf2sG/GebOAZOKuir";
    sout << "RLVHSoehVin2xRsNs6CKF7DhQbxSBrsFq+C9i+1Wvmm9TDtpxcnjhnIrvd/72+F2ltTNBXVls16f";
    sout << "WDSbZtuATt2VQzq5CfnGlu6u+1VPogAt7LeucGmpQkcfPwL41MysG7DIvs0Wn3eytTcb9bmez/xp";
    sout << "IxnqFH+fFicLyEt+Cci+uHzuq8TOLXGUoIksdL4yXtNUmGwgq0cIFa6tafOjRe7Zb4wI25K3V6dt";
    sout << "KxEiyMq1iLSwbEiHZoQg5KMLnhFpRpdk3GEqkjJW9V+r4WV9sR3gPgOtD2HpUfBw3z2Y";

    // Put the data into the istream sin
    sin.str(sout.str());
    sout.str("");

    // Decode the base64 text into its compressed binary form
    base64_coder.decode(sin, sout);
    sin.clear();
    sin.str(sout.str());
    sout.str("");

    // Decompress the data into its original form
    compressor.decompress(sin, sout);

    // Return the decoded and decompressed data
    return sout.str();
}

#endif // SLM_DATA_H
