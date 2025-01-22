#ifndef SlmData_H
#define SlmData_H

#include <string>
#include <vector>
#include <algorithm>

// Utility function to concatenate text parts
inline std::string concatenateTexts(const std::vector<std::string>& texts) {
    std::string result;
    for (const auto& text : texts) {
        result += text;
    }
    return result;
}

// Text parts for training
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

#endif // SlmData_H
